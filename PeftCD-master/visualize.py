import os
import sys
import cv2
import numpy as np
import random
import argparse


def visualize_samples(dataset_dir, pred_dir, output_dir, num_samples=10, split='test'):
    os.makedirs(output_dir, exist_ok=True)

    txt_path = os.path.join(dataset_dir, f'{split}.txt')
    lines = open(txt_path, 'r').readlines()

    pred_files = set(os.listdir(pred_dir))

    samples = random.sample(lines, min(num_samples, len(lines)))

    for i, line in enumerate(samples):
        pathA, pathB, pathLab = line.strip().split('  ')
        pathA = os.path.join(dataset_dir, pathA) if not os.path.isabs(pathA) else pathA
        pathB = os.path.join(dataset_dir, pathB) if not os.path.isabs(pathB) else pathB
        pathLab = os.path.join(dataset_dir, pathLab) if not os.path.isabs(pathLab) else pathLab

        basename = os.path.basename(pathA)
        if basename not in pred_files:
            print(f"Skipping {basename} — no prediction found")
            continue

        imgA = cv2.imread(pathA)
        imgB = cv2.imread(pathB)
        gt = cv2.imread(pathLab, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(pred_dir, basename), cv2.IMREAD_GRAYSCALE)

        if imgA is None or imgB is None or gt is None or pred is None:
            print(f"Skipping {basename} — failed to read")
            continue

        h, w = imgA.shape[:2]

        gt_color = np.zeros((h, w, 3), dtype=np.uint8)
        gt_color[gt > 127] = [0, 255, 0]

        pred_color = np.zeros((h, w, 3), dtype=np.uint8)
        pred_color[pred > 127] = [0, 0, 255]

        overlay_gt = cv2.addWeighted(imgA, 0.6, gt_color, 0.4, 0)
        overlay_pred = cv2.addWeighted(imgA, 0.6, pred_color, 0.4, 0)

        label_h = 30
        canvas_h = h + label_h
        canvas = np.ones((canvas_h, w * 4 + 30, 3), dtype=np.uint8) * 255

        labels = ["Time 1", "Time 2", "Ground Truth", "Prediction"]
        for j, (img, label) in enumerate(zip(
            [imgA, imgB, overlay_gt, overlay_pred], labels
        )):
            x_off = j * (w + 10)
            canvas[label_h:label_h + h, x_off:x_off + w] = img
            cv2.putText(canvas, label, (x_off + 5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out_path = os.path.join(output_dir, f'vis_{i:03d}_{basename}')
        cv2.imwrite(out_path, canvas)
        print(f"Saved: {out_path}")

    print(f"\nDone. {len(samples)} visualizations saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='SYSU-CD')
    parser.add_argument('--pred_dir', type=str,
                        default='work_dirs/DINO3CD_DPT_SYSU_lora_TrainingFiles/test_results')
    parser.add_argument('--output_dir', type=str, default='visualizations')
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()

    dataset_dir = args.dataset
    if not os.path.exists(dataset_dir):
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset)

    visualize_samples(dataset_dir, args.pred_dir, args.output_dir, args.num_samples)
