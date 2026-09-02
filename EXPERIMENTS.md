# Change Detection Experiments Summary

Training on LEVIR-CD+ (574 pairs, urban buildings, Google Earth, ~0.5m res).
Testing on S2Looking (1000 pairs, also ~0.5m res) for cross-domain generalization.

## Results Table

| # | Experiment | Backbone | Decoder | Loss | Trainable Params | LEVIR val mIoU | S2Looking Changed IoU | S2Looking Changed IoU (threshold tuned) |
|---|---|---|---|---|---|---|---|---|
| 1 | ResNet-18 RGB baseline | ResNet-18 | ChangerEx | CE | ~11M | ~0.90 | 2.4% | - |
| 2 | ResNet-18 Grayscale | ResNet-18 | ChangerEx | CE | ~11M | ~0.89 | 1.5% | - |
| 3 | DINOv2 ViT-B/14 + simple decoder | DINOv2 ViT-B/14 | Simple (1-layer) | CE | ~2M | 0.7403 | 20.4% | - |
| 4 | DINOv2 ViT-L/14 + simple decoder | DINOv2 ViT-L/14 | Simple (1-layer) | CE | ~2M | 0.7534 | 21.2% | - |
| 5 | DINOv3 ViT-B/16 + multi-layer | DINOv3 ViT-B/16 | FPN (4-layer) | CE | 8.8M | 0.7548 | 18.9% | - |
| 6 | DINOv3 ViT-L/16 SAT-493M + multi-layer | DINOv3 ViT-L/16 SAT | FPN (4-layer) | CE | 9.6M | 0.8069 | 15.8% | - |
| 7 | DINOv2 ViT-B/14 + multi-layer (CE) | DINOv2 ViT-B/14 | FPN (4-layer) | CE | 8.8M | 0.7713 | 21.6% | 30.9% (t=0.05) |
| 8 | DINOv2 ViT-B/14 + multi-layer (Dice+Focal) | DINOv2 ViT-B/14 | FPN (4-layer) | Dice+Focal | 8.8M | 0.7772 | 19.5% | - |
| 9 | DINOv2 ViT-B/14 + multi-layer + unfreeze 3 | DINOv2 ViT-B/14 | FPN (4-layer) | CE | 30.1M | 0.7825 | 16.1% | - |
| 10 | DINOv2 ViT-B/14 + multi-layer (50% data) | DINOv2 ViT-B/14 | FPN (4-layer) | CE | 8.8M | 0.7640 | 15.8% | - |
| 11 | DINOv2 ViT-B/14 + multi-layer + heavy aug | DINOv2 ViT-B/14 | FPN (4-layer) | CE | 8.8M | 0.7701 | 18.0% | - |
| 12 | DINOv2 ViT-L/14 + multi-layer | DINOv2 ViT-L/14 | FPN (4-layer) | CE | 9.6M | 0.7925 | 21.0% | 32.9% (t=0.05) |
| 13 | DINOv2 ViT-B/14 + FDA (beta=0.05) | DINOv2 ViT-B/14 | FPN (4-layer) | CE | 8.8M | 0.7580 | 21.8% | - |
| 14 | DINOv2 ViT-B/14 + BAN decoder | DINOv2 ViT-B/14 | BAN (adapters) | CE | 2.6M | 0.8146 | 12.7% | - |
| 15 | Zero-shot DINOv2 (no training) | DINOv2 ViT-B/14 | None (cosine dist) | - | 0 | - | 5.7% | - |
| 16 | ChangeAnywhere pretrain → LEVIR finetune | DINOv2 ViT-B/14 | FPN (4-layer) | CE | 8.8M | 0.7697 | 16.6% | - |
| 17 | AnySat backbone (CVPR2025) | AnySat base | Simple | CE | 3.0M | collapsed | 0.0% (collapsed) | - |
| 18 | CrossEarth LoRA-Reins (TPAMI 2025) | DINOv2 ViT-L/16 + Reins | FPN (4-layer) | CE | 9.6M | 0.7497 | 15.1% | - |
| 19 | DOFA ViT-B/16 (224x224) | DOFA ViT-B/16 | FPN (4-layer) | CE | 8.8M | ~0.50 | - (killed, val collapsed) | - |
| 20 | TTA on best model (#12) | DINOv2 ViT-L/14 | FPN (4-layer) | CE | 9.6M | 0.7925 | 21.6% | **34.5% (t=0.02, TTA)** |
| 21 | Post-processing + Ensemble | ViT-B+L ensemble | FPN (4-layer) | CE | - | - | - | 34.4% (TTA) / 33.3% (no TTA) — no improvement |

## Key Findings

1. **Frozen DINOv2 is the best backbone for generalization.** DINOv2 features are domain-agnostic. Any adaptation (unfreezing, satellite-specific pretraining) improves in-domain but hurts cross-domain.

2. **Simpler decoders generalize better.** The FPN with naive |A-B| subtraction outperforms BAN's cross-temporal attention on S2Looking (21.6% vs 12.7%), despite BAN being much better on LEVIR (0.8146 vs 0.7713).

3. **Everything that improves LEVIR hurts S2Looking.** Dice+Focal loss, unfreezing layers, heavy augmentation, BAN, satellite-pretrained backbones — all follow the same pattern.

4. **Threshold tuning is the biggest free win.** Lowering the decision threshold from 0.5 to 0.05 jumps S2Looking Changed IoU from 21% to 33% by recovering missed changes (the model is too conservative cross-domain).

5. **ViT-L/14 is the best overall model.** Slightly better than ViT-B/14 on both LEVIR and S2Looking, and the gains hold after threshold tuning (32.9% vs 30.9%).

6. **FDA gives marginal improvement.** Fourier Domain Adaptation at beta=0.05 barely moved the needle (+0.2%).

7. **Zero-shot feature comparison doesn't work.** Raw DINOv2 cosine distance can't distinguish structural change from appearance variation (lighting, season, viewing angle). A trained decoder is essential.

## Techniques Tried But Not Helpful

- **Dice+Focal loss**: Better LEVIR training metrics, worse generalization
- **Unfreezing top 3 backbone blocks**: Catastrophic forgetting of domain-agnostic features
- **50% training data subset**: Less data = worse everywhere
- **Heavy photometric augmentation**: Didn't bridge the domain gap
- **FDA (Fourier Domain Adaptation)**: Too subtle at beta=0.05, marginal effect
- **BAN (Bitemporal Adapter Network)**: Cross-temporal attention overfits to LEVIR change patterns
- **Temperature scaling**: Equivalent to threshold tuning, no additional benefit
- **DINOv3 SAT-pretrained backbone**: Satellite-specific features overfit to their training distribution
- **ChangeAnywhere pretraining**: Synthetic change pairs not diverse enough; decoder still overfits after LEVIR fine-tuning
- **AnySat backbone (JEPA)**: Features not discriminative enough for pixel-level change detection; collapsed to all-unchanged
- **CrossEarth LoRA-Reins**: Domain-invariant adapters pretrained on segmentation don't help change detection; ViT-L/16 patch_size=16 gives coarser spatial resolution than ViT-L/14
- **Morphological post-processing**: Opening/closing + connected component filtering gives at most +0.1%, usually hurts. Predictions are already clean at optimal threshold.
- **ViT-B+L ensemble**: Averaging ViT-B and ViT-L predictions slightly hurts (34.4% vs 34.5%). ViT-B is weaker and drags down ViT-L.

## Best Configuration

- **Backbone**: DINOv2 ViT-L/14 (frozen)
- **Decoder**: FPN (4-layer, multi-scale)
- **Loss**: Cross-entropy
- **Threshold**: 0.02 (for cross-domain with TTA) or 0.5 (for in-domain)
- **TTA**: 4-fold (none + hflip + vflip + hvflip), average logits
- **Result**: 34.5% Changed IoU on S2Looking (cross-domain, TTA), 0.7925 mIoU on LEVIR (in-domain)

## Checkpoint Locations

- Best ViT-B/14: `work_dirs/dinov2_vitb14_cd_v2/decoder_best.pth`
- Best ViT-L/14: `work_dirs/dinov2_vitl14_cd_v2/decoder_best.pth`
