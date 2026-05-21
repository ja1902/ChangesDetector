import torch
import numpy as np
import cv2
import os
import argparse
from PIL import Image
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
import ModelFactory

torch.set_float32_matmul_precision('high')


def load_model(checkpoint_path, model_name='DINO3CD', peft_method='lora'):
    args = argparse.Namespace(
        model_name=model_name, peft_method=peft_method,
        num_classes=2, loss_type='ce', loss_weights=[1.0, 1.0],
        lr=0.0003, min_lr=0.00003, warmup=3000, max_steps=15000,
        optimizer='adamw', crop_size=256, src_size=256, resize_size=1,
        overlap=128, pred_idx=0, work_dirs='work_dirs',
        exp_name='infer', save_test_results='test_results', model_type='cd',
    )
    model = ModelFactory.SEED_PEFT.load_from_checkpoint(checkpoint_path, args=args)
    model.eval()
    model.cuda()
    return model


def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    img = np.asarray(img)
    transform = Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return transform(image=img)['image'], img


def run_inference(model, tensorA, tensorB, h, w, tile_size=256, overlap=128):
    if h <= tile_size and w <= tile_size:
        inp = torch.cat([tensorA, tensorB], dim=0).unsqueeze(0).cuda()
        with torch.no_grad():
            outs = model(inp)
            logits = (outs[0] + outs[1]) / 2.0
        return logits
    else:
        stride = tile_size - overlap
        pad_h = (stride - (h - tile_size) % stride) % stride
        pad_w = (stride - (w - tile_size) % stride) % stride
        tA = torch.nn.functional.pad(tensorA, (0, pad_w, 0, pad_h))
        tB = torch.nn.functional.pad(tensorB, (0, pad_w, 0, pad_h))
        ph, pw = tA.shape[1], tA.shape[2]

        pred_acc = torch.zeros(1, 2, ph, pw).cuda()
        count = torch.zeros(1, 1, ph, pw).cuda()

        for y in range(0, ph - tile_size + 1, stride):
            for x in range(0, pw - tile_size + 1, stride):
                tileA = tA[:, y:y+tile_size, x:x+tile_size].unsqueeze(0).cuda()
                tileB = tB[:, y:y+tile_size, x:x+tile_size].unsqueeze(0).cuda()
                inp = torch.cat([tileA, tileB], dim=1)
                with torch.no_grad():
                    outs = model(inp)
                    logits = (outs[0] + outs[1]) / 2.0
                pred_acc[:, :, y:y+tile_size, x:x+tile_size] += logits
                count[:, :, y:y+tile_size, x:x+tile_size] += 1

        pred_acc /= count
        return pred_acc[:, :, :h, :w]


def make_comparison(rawA, rawB, gt, prob_map, thresholds, tile_size, overlap, output_path):
    h, w = rawA.shape[:2]
    n_thresh = len(thresholds)

    label_h = 30
    col_w = w
    n_cols = 3 + n_thresh
    canvas_w = n_cols * col_w + (n_cols - 1) * 10
    canvas = np.ones((h + label_h, canvas_w, 3), dtype=np.uint8) * 255

    bgrA = cv2.cvtColor(rawA, cv2.COLOR_RGB2BGR)
    bgrB = cv2.cvtColor(rawB, cv2.COLOR_RGB2BGR)

    gt_color = np.zeros((h, w, 3), dtype=np.uint8)
    gt_color[gt > 127] = [0, 255, 0]
    overlay_gt = cv2.addWeighted(bgrA, 0.6, gt_color, 0.4, 0)

    images = [bgrA, bgrB, overlay_gt]
    labels = ["Before", "After", "Ground Truth"]

    for t in thresholds:
        pred = (prob_map > t).astype(np.uint8)
        pred_color = np.zeros((h, w, 3), dtype=np.uint8)
        pred_color[pred == 1] = [0, 0, 255]
        overlay = cv2.addWeighted(bgrA, 0.6, pred_color, 0.4, 0)
        images.append(overlay)
        labels.append(f"t={t:.2f}")

    for j, (img, label) in enumerate(zip(images, labels)):
        x_off = j * (col_w + 10)
        canvas[label_h:label_h + h, x_off:x_off + col_w] = img
        cv2.putText(canvas, label, (x_off + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    title = f"tile={tile_size} overlap={overlap}"
    cv2.putText(canvas, title, (canvas_w - 300, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    cv2.imwrite(output_path, canvas)
    print(f"Saved: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=str, required=True)
    parser.add_argument('--after', type=str, required=True)
    parser.add_argument('--gt', type=str, default=None, help='Ground truth label (optional)')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='param_sweep_results')
    parser.add_argument('--tile_sizes', type=int, nargs='+', default=[256])
    parser.add_argument('--overlaps', type=int, nargs='+', default=[0])
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.6, 0.7, 0.8, 0.9])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model = load_model(args.checkpoint)

    tensorA, rawA = preprocess(args.before)
    tensorB, rawB = preprocess(args.after)
    h, w = tensorA.shape[1], tensorA.shape[2]

    if args.gt:
        gt = cv2.imread(args.gt, cv2.IMREAD_GRAYSCALE)
    else:
        gt = np.zeros((h, w), dtype=np.uint8)

    for tile_size in args.tile_sizes:
        for overlap in args.overlaps:
            actual_overlap = min(overlap, tile_size // 2)
            print(f"Running: tile_size={tile_size}, overlap={actual_overlap}")
            logits = run_inference(model, tensorA, tensorB, h, w, tile_size, actual_overlap)
            probs = torch.softmax(logits.float(), dim=1)[:, 1].squeeze().cpu().numpy()

            out_name = f"sweep_tile{tile_size}_overlap{actual_overlap}.png"
            make_comparison(
                rawA, rawB, gt, probs,
                args.thresholds, tile_size, actual_overlap,
                os.path.join(args.output_dir, out_name)
            )

    print(f"\nDone. Results in {args.output_dir}/")
