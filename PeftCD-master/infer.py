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
        model_name=model_name,
        peft_method=peft_method,
        num_classes=2,
        loss_type='ce',
        loss_weights=[1.0, 1.0],
        lr=0.0003,
        min_lr=0.00003,
        warmup=3000,
        max_steps=15000,
        optimizer='adamw',
        crop_size=256,
        src_size=256,
        resize_size=1,
        overlap=128,
        pred_idx=0,
        work_dirs='work_dirs',
        exp_name='infer',
        save_test_results='test_results',
        model_type='cd',
    )

    model = ModelFactory.SEED_PEFT.load_from_checkpoint(checkpoint_path, args=args)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, device


def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    img = np.asarray(img)
    transform = Compose([
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    result = transform(image=img)
    return result['image'], img


def run_model_tiled(model, tensorA, tensorB, h, w, device, tile_size=256, overlap=128):
    if h <= tile_size and w <= tile_size:
        inp = torch.cat([tensorA, tensorB], dim=0).unsqueeze(0).to(device)
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

        pred_acc = torch.zeros(1, 2, ph, pw).to(device)
        count = torch.zeros(1, 1, ph, pw).to(device)

        for y in range(0, ph - tile_size + 1, stride):
            for x in range(0, pw - tile_size + 1, stride):
                tileA = tA[:, y:y+tile_size, x:x+tile_size].unsqueeze(0).to(device)
                tileB = tB[:, y:y+tile_size, x:x+tile_size].unsqueeze(0).to(device)
                inp = torch.cat([tileA, tileB], dim=1)
                with torch.no_grad():
                    outs = model(inp)
                    logits = (outs[0] + outs[1]) / 2.0
                pred_acc[:, :, y:y+tile_size, x:x+tile_size] += logits
                count[:, :, y:y+tile_size, x:x+tile_size] += 1

        pred_acc /= count
        return pred_acc[:, :, :h, :w]


def predict_pair(models, imgA_path, imgB_path, output_path='change_map.png',
                 tile_size=256, overlap=128, weights=None):
    tensorA, rawA = preprocess(imgA_path)
    tensorB, rawB = preprocess(imgB_path)
    h, w = tensorA.shape[1], tensorA.shape[2]

    if weights is None:
        weights = [1.0] * len(models)
    total_weight = sum(weights)

    ensemble_logits = None
    for (model, device), weight in zip(models, weights):
        logits = run_model_tiled(model, tensorA, tensorB, h, w, device, tile_size, overlap)
        if ensemble_logits is None:
            ensemble_logits = logits * weight
        else:
            ensemble_logits += logits * weight
    ensemble_logits /= total_weight

    pred = ensemble_logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)

    mask_binary = (pred * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask_binary)
    print(f"Change mask saved: {output_path}")

    overlay_color = np.zeros_like(rawA)
    overlay_color[pred == 1] = [0, 0, 255]
    overlayA = cv2.addWeighted(cv2.cvtColor(rawA, cv2.COLOR_RGB2BGR), 0.6, overlay_color, 0.4, 0)
    overlayB = cv2.addWeighted(cv2.cvtColor(rawB, cv2.COLOR_RGB2BGR), 0.6, overlay_color, 0.4, 0)

    label_h = 30
    canvas = np.ones((h + label_h, w * 3 + 20, 3), dtype=np.uint8) * 255
    for j, (img, label) in enumerate(zip(
        [cv2.cvtColor(rawA, cv2.COLOR_RGB2BGR), cv2.cvtColor(rawB, cv2.COLOR_RGB2BGR), overlayA],
        ["Before", "After", "Changes (red)"]
    )):
        x_off = j * (w + 10)
        canvas[label_h:label_h + h, x_off:x_off + w] = img
        cv2.putText(canvas, label, (x_off + 5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    vis_path = output_path.replace('.png', '_vis.png')
    cv2.imwrite(vis_path, canvas)
    print(f"Visualization saved: {vis_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', type=str, required=True, help='Path to before image')
    parser.add_argument('--after', type=str, required=True, help='Path to after image')
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                        help='Path(s) to model checkpoints. Pass multiple for ensemble.')
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                        help='Weight for each checkpoint (default: equal weights)')
    parser.add_argument('--output', type=str, default='change_map.png')
    parser.add_argument('--model_name', type=str, default='DINO3CD')
    parser.add_argument('--peft_method', type=str, default='lora')
    args = parser.parse_args()

    models = []
    for ckpt in args.checkpoints:
        print(f"Loading model: {ckpt}")
        model, device = load_model(ckpt, args.model_name, args.peft_method)
        models.append((model, device))

    predict_pair(models, args.before, args.after, args.output, weights=args.weights)
