## Prerequisites

- **GPU**: An NVIDIA GPU with at least 4 GB VRAM (8 GB+ recommended)
- **Pretrained weights**: The file `ChangerEx_r18-512x512_40k_levircd.pth` must exist in the project root (downloaded automatically by `install.sh`)
- **Separate virtual environment**: Fine-tuning requires [OpenCD](https://github.com/likyoo/open-cd) and MMEngine, which need Python 3.10-3.12. OpenCD is included in the `open-cd-main/` directory. Create a separate venv for finetuning (this does not affect the main plugin venv):
  ```bash
  python3.10 -m venv venv-finetune
  source venv-finetune/bin/activate
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install mmengine "mmcv-lite>=2.1,<2.2" mmsegmentation ftfy regex
  pip install -e open-cd-main/
  ```
  The main plugin venv is used only for inference. The finetuning venv is used only for training. The resulting `.pth` checkpoint works in either.

## Dataset Preparation

Your dataset must follow this folder structure:

```
my-dataset/
  A/              # "Before" images (PNG, 8-bit RGB)
  B/              # "After" images (PNG, 8-bit RGB)
  label/          # Binary change masks (PNG: 0 = unchanged, 255 = changed)
```

**Requirements:**
- All folders must contain the same filenames (e.g., `001.png` must exist in A/, B/, and label/)
- Images should be PNG format, 8-bit RGB
- Labels must be binary: pixel value 0 for unchanged areas, 255 for changed areas
- Recommended image size: 256x256 or 512x512 patches. If your images are larger, pre-tile them

**Pre-split datasets** are also supported. If you've already split your data, organize it as:

```
my-dataset/
  train/
    A/
    B/
    label/
  val/
    A/
    B/
    label/
```

If no `train/` subdirectory exists, the script will automatically split your data into training and validation sets (80/20 by default).

## Quick Start

```bash
python finetune.py --dataset /path/to/my-dataset
```

That's it. The script will:

1. Validate your dataset structure
2. Auto-split into train/val if needed (using symlinks, your original files are untouched)
3. Generate the training configuration
4. Fine-tune from the pretrained LEVIR-CD checkpoint
5. Save checkpoints and print the path to the best one

## Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | *(required)* | Path to your dataset root |
| `--weights` | `ChangerEx_r18-512x512_40k_levircd.pth` | Pretrained checkpoint to fine-tune from |
| `--lr` | `0.001` | Learning rate |
| `--batch-size` | `8` | Training batch size (reduce if running out of GPU memory) |
| `--iters` | `10000` | Total training iterations |
| `--val-interval` | `1000` | Run validation every N iterations |
| `--val-split` | `0.2` | Fraction of data used for validation when auto-splitting |
| `--grayscale` | off | Convert all images to grayscale (see below) |
| `--work-dir` | auto-generated | Output directory for checkpoints and logs |
| `--amp` | off | Enable mixed-precision training (faster, uses less VRAM) |
| `--resume` | off | Resume from the latest checkpoint in work-dir |

## Grayscale Mode

```bash
python finetune.py --dataset /path/to/my-dataset --grayscale
```

When `--grayscale` is enabled, all images are converted to grayscale during training. This removes color information entirely, forcing the model to rely on structural and textural features for change detection.

**When to use grayscale:**
- Your target imagery comes from a different sensor than the training data
- You're working across regions with very different seasonal/atmospheric conditions
- Color differences between image pairs are causing false positives

**Tradeoffs:**
- Eliminates the color domain gap between sensors, seasons, and regions
- Loses color-based change cues (e.g., vegetation turning to concrete may be harder to detect if their grayscale intensities are similar)

When running inference with a grayscale-trained model, always use the `--grayscale` flag:

```bash
python detect_changes.py --before A.tif --after B.tif --weights <checkpoint> --grayscale
```

## Using Your Fine-tuned Model

### Command line

```bash
python detect_changes.py \
    --before /path/to/before.tif \
    --after /path/to/after.tif \
    --weights /path/to/best_mIoU_iter_XXXXX.pth
```

### QGIS plugin

To use your fine-tuned model in the QGIS plugin:

1. Copy your best checkpoint to the project root and name it appropriately (e.g., `ChangerEx_r18_finetuned.pth`)
2. The model will be available in the plugin's model dropdown after adding it to the registry in `uchange_qgis_plugin/model_bridge.py`

## Tips

### Dataset size
- **50-100 image pairs**: Enough for meaningful improvement. Use fewer iterations (5000-10000)
- **200-500 pairs**: Good results. Default settings should work well
- **1000+ pairs**: Consider increasing iterations to 20000-40000

### Learning rate
- The default `0.001` works well for most fine-tuning scenarios
- If the model is overfitting (validation loss rising while training loss drops), try `0.0005` or `0.0001`
- If training is very slow to converge, try `0.002`

### Memory issues
If you run out of GPU memory:
- Reduce batch size: `--batch-size 4` or `--batch-size 2`
- Enable mixed precision: `--amp`
- Both can be combined: `--batch-size 4 --amp`

### Resuming training
If training is interrupted, resume from where it stopped:
```bash
python finetune.py --dataset /path/to/data --work-dir <same-work-dir> --resume
```
