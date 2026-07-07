# Change Detection Model Benchmark Report

## Background

Our QGIS plugin performs binary change detection on 0.5m resolution satellite imagery. The pipeline tiles large rasters into 256x256 patches, runs inference, and polygonises the output for overlay in QGIS.

We originally integrated two models:

- **MambaBCD** — a state-space model (SSM) based on VMamba. Marketed as having linear complexity with respect to sequence length, which theoretically makes it ideal for large images.
- **PeftCD** — a foundation model approach using DINOv3 with LoRA fine-tuning. Leverages pre-trained vision transformer features for strong generalisation.

Both were selected based on their published state-of-the-art claims. However, during deployment we observed high inference times and VRAM usage that limited practical usability, particularly on consumer and mid-range GPUs. This motivated a broader benchmark to find a model better suited to our operational requirements: fast inference, low VRAM, and competitive accuracy.

## Benchmark Setup

- **Dataset**: LEVIR-CD test set — 128 image pairs, 1024x1024 pixels, building change detection
- **GPU**: NVIDIA Tesla T4 (16 GB VRAM)
- **Metrics**: F1-score, IoU, Precision, Recall, Boundary F1 (BF1), Small Object Recall (SmallRec), Inference Time (ms), Peak VRAM (MB)
- **Models tested**: 18 models from three codebases — 16 from Open-CD, plus MambaBCD and PeftCD
- **Inference**: All Open-CD models ran natively on full 1024x1024 images. MambaBCD and PeftCD used 256x256 tiling with batch size 16, matching our production pipeline.

## Key Results

### Accuracy and Efficiency (1024x1024, LEVIR-CD test set)

| Model | F1 | IoU | Precision | Recall | BF1 | Time (ms) | VRAM (MB) |
|---|---|---|---|---|---|---|---|
| **ChangerEx_r18** | **0.9181** | **0.8486** | 0.9287 | 0.9078 | **0.8025** | **59** | **448** |
| CGNet | 0.9210 | 0.8536 | 0.9360 | 0.9064 | 0.8126 | 1779 | 562 |
| BAN_vit-l14 | 0.9192 | 0.8504 | 0.9363 | 0.9027 | 0.8004 | 1743 | 1077 |
| PeftCD | 0.9149 | 0.8431 | 0.8997 | 0.9306 | 0.7845 | 1891 | 4622 |
| MambaBCD | 0.9065 | 0.8289 | 0.9071 | 0.9058 | 0.7924 | 5190 | 6401 |

### Scaling Benchmark (mosaic of LEVIR-CD tiles)

ChangerEx_r18 was additionally tested on larger mosaics to measure how it handles area coverage typical of real-world change detection tasks.

| Image Size | Time (ms) | VRAM (MB) |
|---|---|---|
| 1024x1024 | 66 | 432 |
| 2048x2048 | 230 | 1,596 |
| 4096x4096 | 924 | 6,252 |
| 8192x8192 | OOM | - |

At 4096x4096 (covering ~4 km2 at 0.5m resolution), ChangerEx_r18 completes inference in under 1 second. Most other models either exceeded 60 seconds or ran out of memory at this scale.

## Analysis

### Why ChangerEx_r18 was selected

ChangerEx_r18 is a CNN-based Siamese encoder-decoder using a ResNet-18 backbone with spatial and channel exchange layers. Despite being an older and simpler architecture than MambaBCD or PeftCD, it dominates on the combined accuracy-efficiency tradeoff:

- **F1 = 0.9181** — within 0.3% of the best model (CGNet at 0.9210), and ahead of MambaBCD (0.9065) and close to PeftCD (0.9149)
- **59 ms inference** — 30x faster than PeftCD (1,891 ms) and 88x faster than MambaBCD (5,190 ms)
- **448 MB VRAM** — 10x less than PeftCD (4,622 MB) and 14x less than MambaBCD (6,401 MB)
- **Scales well** — as a fully convolutional CNN, it handles arbitrary image sizes natively. Convolution operations parallelise efficiently on GPUs.

### Why MambaBCD underperformed

MambaBCD uses the VMamba backbone, which processes image tokens through selective state-space model (SSM) layers. While SSMs offer linear theoretical complexity, the sequential nature of the state-space scan cannot be parallelised on GPUs the way convolution or attention can. Each token depends on the hidden state from the previous token, forcing serial execution through the sequence.

The result: **5,190 ms and 6,401 MB at 1024x1024** — the slowest and most memory-hungry model tested. When tested natively (without tiling) at 1024x1024, MambaBCD took over 52 seconds due to the full-length sequential scan. Its F1 (0.9065) was also below average for the benchmark.

SSMs may show advantages on extremely long sequences or in training-time compute, but for GPU inference on image-sized inputs, the lack of parallelism is a significant bottleneck.

### Why PeftCD underperformed

PeftCD uses a DINOv3 vision transformer backbone with LoRA adapters. The LoRA approach is parameter-efficient for fine-tuning (only a small number of adapter weights are trained), but at inference time the full ViT backbone must still be executed. This results in high compute and memory costs:

- **1,891 ms** — 32x slower than ChangerEx, despite using tiled 256x256 inference
- **4,622 MB VRAM** — the large ViT backbone dominates memory even on small tiles
- **F1 = 0.9149** — good accuracy, particularly with the highest recall (0.9306) of any model, but the efficiency penalty is severe for our use case

PeftCD's strength lies in generalisation from pre-trained features, which may be more valuable when fine-tuning data is scarce. For our case with established training datasets, the accuracy gain does not justify the 32x slowdown.

## Conclusion

ChangerEx_r18 has been integrated into the ChangeDetection QGIS plugin as a direct replacement for MambaBCD. MambaBCD offers no advantage — ChangerEx matches or exceeds its accuracy while being 88x faster and using 14x less VRAM. There is no scenario in our pipeline where MambaBCD is the better choice.

PeftCD is a different case. Its foundation model approach provides stronger generalisation from pre-trained features, which matters when applying the model to imagery or change types it was not explicitly trained on. Until we can build a custom training dataset for our specific use case (0.5m imagery, target change types, geographic region), PeftCD remains valuable as a fallback for out-of-distribution scenarios where ChangerEx's accuracy may degrade. Once a custom dataset is available and ChangerEx can be fine-tuned on domain-specific data, it would likely outperform PeftCD on that domain while retaining its efficiency advantage — at which point PeftCD would no longer be needed.
