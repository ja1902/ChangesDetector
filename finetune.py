"""
Fine-tune ChangerEx on a custom change detection dataset.

Usage:
    python finetune.py --dataset /path/to/my-data
    python finetune.py --dataset /path/to/my-data --grayscale --iters 20000
    python finetune.py --dataset /path/to/my-data --lr 0.0005 --batch-size 4 --amp

See FINETUNING.md for full documentation.
"""

import sys
import os
import argparse
import glob
import random
import textwrap
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPENCD_ROOT = os.path.join(SCRIPT_DIR, "open-cd-main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune ChangerEx on a custom dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Dataset format:
              Your dataset must follow the LEVIR-CD folder structure:
                dataset/
                  A/         (before images, PNG)
                  B/         (after images, PNG)
                  label/     (binary change masks, PNG: 0=unchanged, 255=changed)

              If train/ and val/ subdirectories exist, they are used directly.
              Otherwise the dataset is automatically split (see --val-split).
        """),
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to dataset root (must contain A/, B/, label/ folders)")
    parser.add_argument(
        "--weights", default=os.path.join(SCRIPT_DIR, "ChangerEx_r18-512x512_40k_levircd.pth"),
        help="Pretrained checkpoint to fine-tune from (default: project's LEVIR-CD weights)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size (default: 8)")
    parser.add_argument("--iters", type=int, default=10000, help="Total training iterations (default: 10000)")
    parser.add_argument("--val-interval", type=int, default=1000, help="Validate every N iterations (default: 1000)")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio for auto-split (default: 0.2)")
    parser.add_argument("--grayscale", action="store_true", help="Convert images to grayscale during training and inference")
    parser.add_argument("--work-dir", default=None, help="Output directory for checkpoints and logs")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed-precision training")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint in work-dir")
    return parser.parse_args()


def validate_dataset(dataset_path):
    """Check dataset structure and return (data_root, is_pre_split)."""
    if not os.path.isdir(dataset_path):
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    train_dir = os.path.join(dataset_path, "train")
    if os.path.isdir(train_dir) and os.path.isdir(os.path.join(train_dir, "A")):
        for split in ("train", "val"):
            split_dir = os.path.join(dataset_path, split)
            if not os.path.isdir(split_dir):
                print(f"ERROR: Pre-split dataset missing '{split}/' directory")
                sys.exit(1)
            _check_abc_dirs(split_dir, split)
        return dataset_path, True

    _check_abc_dirs(dataset_path, "dataset root")
    return dataset_path, False


def _check_abc_dirs(path, label):
    """Verify A/, B/, label/ exist and have matching files."""
    for subdir in ("A", "B", "label"):
        d = os.path.join(path, subdir)
        if not os.path.isdir(d):
            print(f"ERROR: {label} is missing '{subdir}/' directory (expected at {d})")
            sys.exit(1)

    a_files = set(_list_images(os.path.join(path, "A")))
    b_files = set(_list_images(os.path.join(path, "B")))
    label_files = set(_list_images(os.path.join(path, "label")))

    if not a_files:
        print(f"ERROR: No PNG images found in {os.path.join(path, 'A')}")
        sys.exit(1)

    if a_files != b_files:
        missing_in_b = a_files - b_files
        missing_in_a = b_files - a_files
        msg = f"ERROR: Filename mismatch between A/ and B/ in {label}."
        if missing_in_b:
            msg += f"\n  In A/ but not B/: {sorted(missing_in_b)[:5]}"
        if missing_in_a:
            msg += f"\n  In B/ but not A/: {sorted(missing_in_a)[:5]}"
        print(msg)
        sys.exit(1)

    if a_files != label_files:
        missing = a_files - label_files
        extra = label_files - a_files
        msg = f"ERROR: Filename mismatch between A/ and label/ in {label}."
        if missing:
            msg += f"\n  In A/ but not label/: {sorted(missing)[:5]}"
        if extra:
            msg += f"\n  In label/ but not A/: {sorted(extra)[:5]}"
        print(msg)
        sys.exit(1)

    print(f"  {label}: {len(a_files)} image pairs OK")


def _list_images(directory):
    """List image basenames (without path) in a directory."""
    return [
        os.path.basename(f)
        for f in glob.glob(os.path.join(directory, "*.png"))
    ]


def auto_split(dataset_path, val_ratio, seed=42):
    """Split a flat dataset into train/val using symlinks."""
    filenames = sorted(_list_images(os.path.join(dataset_path, "A")))
    random.seed(seed)
    random.shuffle(filenames)

    n_val = max(1, int(len(filenames) * val_ratio))
    val_files = set(filenames[:n_val])
    train_files = set(filenames[n_val:])

    split_root = os.path.join(dataset_path, "_split")
    if os.path.exists(split_root):
        print(f"  Auto-split directory already exists: {split_root}")
        print("  Reusing existing split.")
        return split_root

    for split_name, file_set in [("train", train_files), ("val", val_files)]:
        for subdir in ("A", "B", "label"):
            dest_dir = os.path.join(split_root, split_name, subdir)
            os.makedirs(dest_dir, exist_ok=True)
            src_dir = os.path.join(dataset_path, subdir)
            for fname in file_set:
                src = os.path.abspath(os.path.join(src_dir, fname))
                dst = os.path.join(dest_dir, fname)
                os.symlink(src, dst)

    print(f"  Auto-split: {len(train_files)} train, {len(val_files)} val")
    print(f"  Split saved to: {split_root}")
    return split_root


def generate_config(args, data_root, work_dir):
    """Generate a temporary MMEngine config file for fine-tuning."""
    if args.grayscale:
        base_config = os.path.join(OPENCD_ROOT, "configs/changer/changer_ex_r18_512x512_40k_levircd_gray.py")
    else:
        base_config = os.path.join(OPENCD_ROOT, "configs/changer/changer_ex_r18_512x512_40k_levircd.py")

    weights_path = os.path.abspath(args.weights)
    abs_data_root = os.path.abspath(data_root)

    config_content = textwrap.dedent(f"""\
        _base_ = '{base_config}'

        load_from = '{weights_path}'

        data_root = '{abs_data_root}'

        train_dataloader = dict(
            batch_size={args.batch_size},
            dataset=dict(
                data_root=data_root,
                data_prefix=dict(
                    img_path_from='train/A',
                    img_path_to='train/B',
                    seg_map_path='train/label')))

        val_dataloader = dict(
            dataset=dict(
                data_root=data_root,
                data_prefix=dict(
                    img_path_from='val/A',
                    img_path_to='val/B',
                    seg_map_path='val/label')))

        test_dataloader = val_dataloader

        optimizer = dict(
            type='AdamW', lr={args.lr}, betas=(0.9, 0.999), weight_decay=0.05)
        optim_wrapper = dict(
            _delete_=True,
            type='OptimWrapper',
            optimizer=optimizer)

        train_cfg = dict(type='IterBasedTrainLoop', max_iters={args.iters}, val_interval={args.val_interval})

        default_hooks = dict(
            checkpoint=dict(type='CheckpointHook', by_epoch=False,
                            interval={args.val_interval}, save_best='mIoU'))
    """)

    os.makedirs(work_dir, exist_ok=True)
    config_path = os.path.join(work_dir, "finetune_config.py")
    with open(config_path, "w") as f:
        f.write(config_content)

    return config_path


def _patch_mmcv_ops():
    """Stub mmcv.ops so mmseg imports work with mmcv-lite (no CUDA ops).

    mmseg unconditionally imports various CUDA ops (point_sample,
    sigmoid_focal_loss, etc.) from mmcv.ops at module load time.
    With mmcv-lite those don't exist. We inject a lazy stub module
    that returns None for any missing attribute, which is enough for
    import to succeed. The ops are never called during ChangerEx training.
    """
    import types
    import mmcv

    class _LazyOpsModule(types.ModuleType):
        def __getattr__(self, name):
            return None

    if not hasattr(mmcv, 'ops') or not hasattr(mmcv.ops, 'point_sample'):
        from importlib.machinery import ModuleSpec
        ops = _LazyOpsModule('mmcv.ops')
        ops.__spec__ = ModuleSpec('mmcv.ops', None)
        sys.modules['mmcv.ops'] = ops
        ext = types.ModuleType('mmcv._ext')
        ext.__spec__ = ModuleSpec('mmcv._ext', None)
        sys.modules['mmcv._ext'] = ext
        mmcv.ops = ops


def run_training(config_path, work_dir, amp, resume):
    """Run the MMEngine training loop."""
    if OPENCD_ROOT not in sys.path:
        sys.path.insert(0, OPENCD_ROOT)

    os.chdir(OPENCD_ROOT)
    _patch_mmcv_ops()

    from mmengine.config import Config
    from mmengine.runner import Runner
    import opencd  # noqa: F401 — registers custom modules with mmengine

    cfg = Config.fromfile(config_path)
    cfg.work_dir = work_dir
    cfg.resume = resume

    if amp:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    runner = Runner.from_cfg(cfg)
    runner.train()


def find_best_checkpoint(work_dir):
    """Find the best checkpoint in work_dir."""
    pattern = os.path.join(work_dir, "best_mIoU_iter_*.pth")
    candidates = glob.glob(pattern)
    if candidates:
        return sorted(candidates)[-1]
    latest = os.path.join(work_dir, "latest.pth")
    if os.path.exists(latest):
        return latest
    return None


def main():
    args = parse_args()

    print("=" * 60)
    print("ChangerEx Fine-tuning")
    print("=" * 60)

    if not os.path.isfile(args.weights):
        print(f"ERROR: Pretrained weights not found: {args.weights}")
        print("Make sure the weights file exists or specify --weights.")
        sys.exit(1)

    print(f"\nDataset: {args.dataset}")
    print("Validating dataset structure...")
    data_root, is_pre_split = validate_dataset(args.dataset)

    if not is_pre_split:
        print(f"\nFlat dataset detected. Auto-splitting ({1 - args.val_split:.0%} train / {args.val_split:.0%} val)...")
        data_root = auto_split(args.dataset, args.val_split)

    if args.work_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.work_dir = os.path.join(SCRIPT_DIR, "open-cd-main", "work_dirs", f"finetune_{timestamp}")

    work_dir = os.path.abspath(args.work_dir)

    print(f"\nConfiguration:")
    print(f"  Pretrained weights: {args.weights}")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Batch size:         {args.batch_size}")
    print(f"  Iterations:         {args.iters}")
    print(f"  Val interval:       {args.val_interval}")
    print(f"  Grayscale:          {args.grayscale}")
    print(f"  Mixed precision:    {args.amp}")
    print(f"  Work directory:     {work_dir}")

    print("\nGenerating training config...")
    config_path = generate_config(args, data_root, work_dir)
    print(f"  Config saved to: {config_path}")

    print("\nStarting training...\n")
    run_training(config_path, work_dir, args.amp, args.resume)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    best_ckpt = find_best_checkpoint(work_dir)
    if best_ckpt:
        print(f"\nBest checkpoint: {best_ckpt}")
        grayscale_flag = " --grayscale" if args.grayscale else ""
        print(f"\nTo run inference with your fine-tuned model:")
        print(f"  python detect_changes.py --before <before.tif> --after <after.tif> \\")
        print(f"      --weights {best_ckpt}{grayscale_flag}")
    else:
        print(f"\nCheckpoints saved in: {work_dir}")


if __name__ == "__main__":
    main()
