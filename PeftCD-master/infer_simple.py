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
        overlap=0, pred_idx=0, work_dirs='work_dirs',
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


def run_inference(model, tensorA, tensorB, h, w, tile_size=256, overlap=0, batch_size=8):
    if h <= tile_size and w <= tile_size:
        inp = torch.cat([tensorA, tensorB], dim=0).unsqueeze(0).cuda()
        with torch.no_grad(), torch.autocast('cuda'):
            outs = model(inp)
            logits = (outs[0] + outs[1]) / 2.0
        return logits

    stride = tile_size - overlap
    pad_h = (stride - (h - tile_size) % stride) % stride
    pad_w = (stride - (w - tile_size) % stride) % stride
    tA = torch.nn.functional.pad(tensorA, (0, pad_w, 0, pad_h)).cuda()
    tB = torch.nn.functional.pad(tensorB, (0, pad_w, 0, pad_h)).cuda()
    ph, pw = tA.shape[1], tA.shape[2]

    pred_acc = torch.zeros(1, 2, ph, pw, device='cuda')
    count = torch.zeros(1, 1, ph, pw, device='cuda')

    coords = [
        (y, x)
        for y in range(0, ph - tile_size + 1, stride)
        for x in range(0, pw - tile_size + 1, stride)
    ]

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i + batch_size]
        batch_A = torch.stack([tA[:, y:y+tile_size, x:x+tile_size] for y, x in batch_coords])
        batch_B = torch.stack([tB[:, y:y+tile_size, x:x+tile_size] for y, x in batch_coords])
        inp = torch.cat([batch_A, batch_B], dim=1)
        with torch.no_grad(), torch.autocast('cuda'):
            outs = model(inp)
            logits = (outs[0] + outs[1]) / 2.0
        for j, (y, x) in enumerate(batch_coords):
            pred_acc[:, :, y:y+tile_size, x:x+tile_size] += logits[j].unsqueeze(0)
            count[:, :, y:y+tile_size, x:x+tile_size] += 1

    pred_acc /= count
    return pred_acc[:, :, :h, :w]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=str, required=True)
    parser.add_argument('--after', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='change_map.png')
    parser.add_argument('--tile_size', type=int, default=256)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    print("Loading model...")
    model = load_model(args.checkpoint)

    tensorA, rawA = preprocess(args.before)
    tensorB, rawB = preprocess(args.after)
    h, w = tensorA.shape[1], tensorA.shape[2]
    if rawB.shape[:2] != (h, w):
        rawB = cv2.resize(rawB, (w, h), interpolation=cv2.INTER_LINEAR)

    print(f"Running inference ({w}x{h}, tile={args.tile_size}, overlap={args.overlap}, threshold={args.threshold})...")
    logits = run_inference(model, tensorA, tensorB, h, w, args.tile_size, args.overlap, args.batch_size)
    probs = torch.softmax(logits.float(), dim=1)[:, 1].squeeze().cpu().numpy()

    binary = (probs > args.threshold).astype(np.uint8)
    change_pct = binary.sum() / binary.size * 100
    print(f"Change pixels: {binary.sum()} / {binary.size} ({change_pct:.2f}%)")

    bgrA = cv2.cvtColor(rawA, cv2.COLOR_RGB2BGR)
    bgrB = cv2.cvtColor(rawB, cv2.COLOR_RGB2BGR)

    pred_color = np.zeros((h, w, 3), dtype=np.uint8)
    pred_color[binary == 1] = [0, 0, 255]
    overlayA = cv2.addWeighted(bgrA, 0.6, pred_color, 0.4, 0)
    overlayB = cv2.addWeighted(bgrB, 0.6, pred_color, 0.4, 0)

    label_h = 30
    canvas = np.ones((h + label_h, w * 3 + 20, 3), dtype=np.uint8) * 255
    for j, (img, label) in enumerate(zip(
        [bgrA, bgrB, overlayA],
        ["Before", "After", f"Changes (t={args.threshold})"]
    )):
        x_off = j * (w + 10)
        canvas[label_h:, x_off:x_off + w] = img
        cv2.putText(canvas, label, (x_off + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imwrite(args.output, canvas)
    print(f"Saved: {args.output}")
