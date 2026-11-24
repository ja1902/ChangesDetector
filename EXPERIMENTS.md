# Change Detection Experiments Summary

Two parts: **Part 1** (Sep 2026) picks a backbone and decoder by training on LEVIR-CD+
and testing on S2Looking. **[Part 2](#part-2-custom-dataset-oct-2026)**
(Oct 2026) builds a custom hand-reviewed dataset from Google Earth Pro and uses
it to choose a fine-tuning recipe.

# Part 1: LEVIR-CD+ → S2Looking (Sep 2026)

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

---

# Part 2: Custom dataset (Oct 2026)

Goal: try fine-tuning techniques on the custom imagery until one clearly wins, then test
the winner on pairs from places it has never seen. Every number below is on squares
held out from training.

## Dataset

13 squares of 1 km², each cut from a pair of Google Earth Pro captures (two dates, same
untouched view, georeferenced by hand):

| batch | squares | pixel size | real changes |
|---|---|---|---|
| first 7 (reviewed by 2026-10-13) | 7 squares in 5 places (two places have two squares each) | 0.38 m | 333 |
| r01–r06 (2026-10-19) | 6 separate places, mostly hilly villages | 0.488 m | 342 |
| **total** | 13 squares, 11 places | | **675** |

Pipeline (all in `benchmarks/`):
1. `prep_pairs.py`: captures + GCP points / world file → georeferenced scene pairs.
2. `crop_pairs.py`: removes the few-metre shift between dates and cuts randomly placed
   squares (random so the test isn't tilted toward easy places; same seed → same squares).
3. `candidates.py`: union of blobs flagged by several detectors (shipped, ensemble,
   fused / nudged; later the fine-tuned ones).
4. `review_pack.py` + `review_style.py`: one `candidates.gpkg` per square, round 1 (usual
   thresholds) and round 2 (sensitive thresholds), styled for QGIS.
5. Review in QGIS: verdict real / false / unsure, plus polygons for missed areas
   (`extra.gpkg` holds a second batch of candidates).
6. `make_labels.py` → `train_label.tif`. Real outlines are tightened with SAM 2 (on the
   after image, the before image for demolitions) when SAM agrees.

| pixel | label | trained on? |
|---|---|---|
| no detector flagged it | 0 | yes |
| candidate marked **false** | 0 | yes |
| candidate **unsure** or unreviewed, tiny set-aside blobs, band around each change | 255 | no |
| candidate marked **real**, or drawn as missed | 1 | yes |

A "false" verdict sets the same label value as background. What it does is allow
training on exactly the spots where models go wrong (seasonal fields, shadows, building
lean, colour shifts); unreviewed candidates have to be ignored. Reviewing took about an
hour per km², which is why each scene contributed only one square. The full scenes
(~111 km² altogether) are mostly unlabelled.

## Evaluation protocol

- **Leave-one-city-out** (`loco_finetune.py`): fine-tune on the other cities, score the
  held-out one. City = first 3 letters of the square name. Never hold out one square of a
  city while its sibling stays in training.
- **Threshold-free comparison** (`compare_loco.py`): pixel AP, best IoU, share of real
  changes found at 20 false alarms/km², and false alarms/km² needed to find 85%.
- **Run-to-run noise:** two identical runs of the plain fine-tune differ by about 0.02
  pooled AP, 2 points of recall and 6–9 FA/km². Per place it is 0.005–0.007 AP for most,
  but one place swings more (0.84 → 0.78 on one square). Anything smaller is
  a tie.

## Technique comparison (7 squares, 333 changes)

All fine-tunes start from the shipped `dinov2_vitb14_c2s1_levir` decoder: 3000 steps,
decoder only, crops half from our squares and half from LEVIR-CD (`finetune_dinov2.py`).
"+ nudge" adds the label-free score with weight 1.0 (see below).

| | pixel AP | best IoU | found at 20 FA/km² | FA/km² to find 85% |
|---|---|---|---|---|
| shipped | 0.599 | 0.452 | 45% | 116 |
| shipped + nudge | 0.640 | 0.498 | 48% | 99 |
| plain fine-tune, run 1 | 0.693 | 0.505 | 54% | 74 |
| plain fine-tune, run 2 | 0.673 | 0.488 | 55% | 82 |
| **plain fine-tune + nudge, run 1** | **0.727** | **0.555** | 62% | 63 |
| plain fine-tune + nudge, run 2 | 0.712 | 0.537 | 61% | 69 |
| plain + nudge weight 0.5 | 0.721 | 0.539 | 61% | 71 |
| hard negatives | 0.666 | 0.481 | 59% | 92 |
| hard negatives + nudge | 0.715 | 0.534 | 62% | 67 |
| 25% LEVIR | 0.688 | 0.499 | 62% | 85 |
| 25% LEVIR + nudge | 0.723 | 0.545 | 63% | 70 |
| average of runs 1+2 | 0.687 | 0.501 | 53% | 76 |
| average of runs 1+2 + nudge | 0.722 | 0.549 | 63% | 63 |
| average of runs 1+2 + nudge weight 1.5 | 0.713 | 0.552 | 54% | 76 |
| unfreeze last 2 backbone blocks | 0.653 | 0.483 | 55% | 81 |
| unfreeze last 2 blocks + nudge | 0.692 | 0.525 | 62% | 62 |
| 6000 steps | stopped after 2 held-out places: 0.757 → 0.698 and 0.433 → 0.382 AP | | | |

Findings:
1. **Fine-tuning on the training data is a clear gain:** AP 0.60 → 0.67–0.69, about 4× the
   run-to-run noise.
2. **The plain recipe is the pick.** Hard negatives, 25% LEVIR and averaging two runs all
   land inside the two plain runs' range, so the simpler recipe wins.
3. **Hard negatives** (30% of crops centred on "false" candidates, their pixels weighted
   ×2) work at the strict end (59% vs 54% found at 20 FA/km²) but overcorrect: many
   false alarms look like early-stage real change (disturbed ground, new paving,
   half-built sites), so high-recall performance and AP drop. A gentler setting
   (15% / ×1.5) is untested and unlikely to beat the noise.
4. **Unfreezing is unstable** (AP on one held-out place fell to 0.45–0.57). **Longer training overfits**
   the training cities and makes the model more cautious in new ones.
5. Nudge weight 1.0 beats 0.5 and 1.5.

## 13 squares (adding r01–r06)

| | pixel AP | best IoU | found at 20 FA/km² | FA/km² to find 85% |
|---|---|---|---|---|
| shipped | 0.614 | 0.459 | 51% | 97 |
| shipped + nudge | 0.655 | 0.496 | 55% | 73 |
| fine-tune on 13 | 0.691 | 0.504 | 66% | 59 |
| **fine-tune on 13 + nudge** | **0.709** | **0.526** | **68%** | **56** |

- Per square (AP, shipped → fine-tune + nudge): big gains on four squares of the
  training data (0.24 → 0.51, 0.51 → 0.66, 0.68 → 0.85, 0.26 → 0.38), smaller gains on
  three (0.64 → 0.80, 0.74 → 0.85, 0.34 → 0.46), about level on the other six.
- **Doubling the data gave no measurable gain on the first 7 squares** (AP 0.713 trained
  on ~10 other places, vs 0.712–0.727 trained on 4). Likely reasons: the new places are
  a different kind of terrain, their imagery is coarser (0.49 m vs 0.38 m), and the
  labels themselves may now be the limit. More squares of the same kind are no longer a
  cheap win; new kinds of place are.

Thresholds for the fine-tune + nudge, pooled over the 13 held-out squares:

| threshold | changes found | FA/km² | precision | F1 |
|---|---|---|---|---|
| 0.90 | 77% | 32 | 54% | 0.633 |
| **0.94** (plugin default) | 72% | 22 | 61% | 0.661 |
| 0.96 (best F1) | 66% | 17 | 67% | 0.665 |
| 0.98 | 59% | 10 | 75% | 0.656 |

Without the nudge, the balanced threshold is **0.85** (F1 0.649). For comparison, the
shipped model at its balanced point finds 64% with 36 FA/km².

On the public test sets, which label new buildings only, the fine-tuned model scores
lower than the shipped one (AP: S2Looking 0.437 vs 0.501, LEVIR-CD 0.952 vs 0.960,
EGY-BCD 0.689 vs 0.778), partly because it now also flags construction sites and
cleared land.

## The label-free nudge

`uchange_qgis_plugin/labelfree.py`, `--label-free W`. No training. For each image pair:
DINOv2 describes every 14×14 patch on both dates; each patch's change is measured
against its 3×3 neighbours on the other date (tolerates misalignment and lean); patches
are grouped into 16 surface types per date; each change is scored by how unusual it is
for its own surface type in this pair (z ≈ 0 normal, > 0 unusual). W·z is added to the
model's logit before thresholding. Seasonal, sensor-colour and haze changes affect a
whole surface type at once, so they look normal. On its own it is a weak detector (it
also flags newly ploughed fields or filled pools). Costs about 11 s per km² on the GPU.

- **Public data (S2Looking):** weight chosen on validation, measured on test: about +6%
  for both models. No label bias here.
- **Our squares:** it raises AP for every model (table above), but polygon counts at the
  recommended thresholds tell a weaker story:

  | 13 held-out squares | polygons | real changes found | false alarms |
  |---|---|---|---|
  | nudge on (0.94) | 746 | 485 / 675 (72%) | 290 |
  | nudge off (0.85) | 694 | 445 / 675 (66%) | 251 |
  | difference | +52 | +40 | +39 |

  Each extra find costs about one extra false polygon. At the same false-alarm rate
  (~19/km²) the nudge finds only about 2% more (458 vs 445). It helps clearly on a few
  squares of the training data and mostly adds noise on others.
- **Label bias in its favour:** some candidates came from nudged detectors (`fused` in the
  first 7 squares, `ftz` in r01–r06), so changes only the nudge found were labelled real.
- **Conclusion (2026-10-20):** once the model is fine-tuned on our imagery, the nudge's real
  benefit is probably close to zero. It mattered when the model hadn't seen this kind of
  imagery. The suggestion is to make **nudge off at 0.85** the default and keep the
  checkbox (not decided yet). On ground the model trained on, the nudge only adds noise
  (78 vs 39 polygons on r05), so compare the two only on new places.

## November 2026: what more can be found (v0.10)

Sixteen ideas were tested against fixed bars, each on squares the model was not trained on
(leave one city out). Score: "curve mean" = share of real changes found, averaged over
5-28 false alarms per km2; a win needed +3 points with a bootstrap interval (over 200 m
cells) clear of zero. Before that, the unjudged detections counted as false alarms were
reviewed blind (about 400 panels): over half were real changes, so the baseline rose.

| idea | result |
|---|---|
| wider tiles (364-742 px), and a wider view for the filter | level |
| per-scale confidences in the filter | level |
| teaching the filter a pair from its sibling square's verdicts | level |
| height from a monocular depth model (Depth Anything V2 Small) | level |
| instance-balanced loss (small changes weigh more in training) | level, inside run-to-run noise |
| small set-aside candidates reviewed and used in training (405 blind panels: 142 real, 250 not) | level: two seeds -2.4 to +0.9 against both baselines; small changes not found more often |
| finer view at inference (2.0x upsampling instead of 1.4x, about 3 m per patch instead of 4.5 m) | worse: -7.8 [-10.2, -4.7] with the filter on 13 squares, small changes too; a +3.5 on 4 squares did not hold |
| averaging the two models' maps (LEVIR-CD and S2Looking partners) | +1.3 over the recommended model but -2.6 [-4.5, -1.1] against the S2Looking model alone, at twice the run time |
| match certainty from a wide-baseline matcher (RoMa) as filter input | level: +0.0 [-1.2, +1.8] (-2.0 on the verdicts before the blind review); the filter's own ranking improved, the end result did not |
| null pairs: 6 SpaceNet-4 views (7-32 deg off nadir) of 200 places, all "no change", as a quarter of the training crops | level: without the filter +9.0 and +8.7 (two seeds; small changes found twice as often), but with the filter -3.5 to +2.4 against the baselines and -4.0 / -7.8 against the S2Looking model. The filter already removes the lean false alarms the pairs teach the model to avoid |
| co-registration (on by default in the plugin; the benchmark maps are made without it) | harmless: +0.9 [-2.1, +3.6] on the training pairs, which are already aligned to half a pixel. With the after image moved 3 / -4 px the model loses only 0.9 without it and co-registration gets back 1.4; it applied the right correction on 7 of 13 squares and declined the rest |
| per-tile histogram matching (the old "Match image histograms" option) | worse: -2.7 [-4.8, -0.3] with the filter, -7.3 without; removed from the dialog |
| null pairs on top of the S2Looking recipe | level against the S2Looking model: without the filter +5.8 and +4.0, with it -2.8 to +2.5 (two seeds against two) |
| **weaker changes judged by the filter** (peaks >= 0.45 outside the detections) | **+3.8** [+2.5, +5.5] on 13 squares, 964 changes; found at 20 FA/km2 73% -> 80%, under 100 m2 191 -> 232 of 335 |
| **S2Looking instead of LEVIR-CD as the training partner** | **+4.0** [+1.7, +6.4] with the filter (threshold only +6.3); by place of the training data, found at 10 FA/km2: 49% -> 72% and 64% -> 76% on two, 73% -> 55% on a third |
| both together | +4.9 [+2.6, +7.8]; only +1.1 over the weaker changes alone: both win the same small changes |

On pr1_c1 (unseen, strong building lean; 31 more blind panels) all four setups are level:
at the default cut 8 (shipped) / 9 (S2Looking) changes with 1 false alarm per km2, at 20
false alarms per km2 14 / 15 (with weaker changes) / 16. On pr2_c1 the weaker changes alter
nothing at the default cut.

Shipped in v0.10: the weaker changes in both finetuned presets (`orphan_peak` 0.45), and the
S2Looking model as a second option (`--preset finetuned_s2l`), not the default because of
that one loss. Its false-alarm filter keeps the shared first stage and refits the second
on its own held-out maps. Retrain it with `finetune_dinov2.py --levir <S2Looking>
--partner-dirs Image1,Image2 --partner-crop 370 --seed 1` on all 13 squares.


## Speed and GPU memory (v0.10.2)

Profiled on pr2_c1 (1 km2 at 0.49 m; GPU: a T4; CPU: 24 threads). GPU run: inference 31 s, of which flip averaging 22 s (98%
of tiles reach the 0.5 gate), the filter 28 s (DINOv2 on its crops 23 s). The decoder costs
as much as the backbone (131 against 134 ms per 8 tiles). Peak 1.04 GB, 0.8 GB of it
decoder feature maps. CPU, a 500 m area: inference 70% (flips three quarters of it), the
filter 30%. Float16 results depend on the batch size (8 against 4 tiles: up to 0.017 in
probability, 5 of 22 outlines move a little); the same run repeated is byte for byte the same.

| idea | result |
|---|---|
| **filter: one rotation first**, the other three only within 1.75 logit of the cut | **kept**: filter 28 -> 18-19 s on the GPU, 86-88 -> 47-49 s on the CPU area; output byte for byte the same. On 15 areas (3,772 candidates) the 1-rotation score is at most 1.21 below the full one (99.9% within 1.01) with either filter |
| **decode a few tiles at a time when memory is short** (keep 8 tiles per batch) | **kept**: 0.8 -> 0.23 GB for 8 tiles at +3% time; 0.4% of map pixels move by 1/255; 13 areas with the filter: curve mean 66.1% both, +0.1 [-0.1, +0.2] |
| **CPU: one thread fewer than cores** | **kept**: 0.76 against 1.15 s per tile with one other busy process; same results |
| **resize the positional embedding once** | **kept**: 9 ms of each 260 ms pass on the T4; same results |
| skip weaker changes that provably cannot reach the cut (the score is linear) | nothing to skip: the tightest exact bound on stage 1 is 102 logit, so only a p90 under 1e-28 is ruled out |
| skip weaker changes by their map p90 | no: p90 barely predicts the score (r = 0.32); p90 < 0.5 drops 24% of them but 21 of the 158 that pass the 20/km2 cut |
| channels_last, cudnn.benchmark | no gain |
| flash attention | not on a T4 (needs sm80); the memory-efficient kernel is already used |
| torch.compile on the decoder | 10% faster, but 16 s to compile in every run, and not exact |
| a higher flip gate | little to gain: 78% of tiles still reach 0.8 on pr2_c1 |
| loading the model during co-registration | dropped: AROSICS's helpers use multiprocessing, and forking while torch loads in another thread can deadlock, for ~1.5 s |
| CPU batch of 8 | -10% to +2% depending on threads; not taken |
| **CPU: decoder one tile at a time** | **kept**: 2.4 against 3.1 s per 4 tiles (the decoder's maps stay in cache); same GeoPackage |
| **OpenVINO 2026.4 on the CPU** (float32, converted straight from PyTorch) | **kept**: against the PyTorch path above, the change model x1.46 and the filter's backbone (280 px crops) x1.20 on a 6-core laptop CPU, x1.19 and x1.40 on a 24-thread one (`--groups`: best at 4 tiles and 16 crops per call); a 500 m area 196-205 s against 220-222 s, GeoPackage byte for byte the same; peak memory +0.7 GB (`benchmarks/cpu_backends.py`) |
| ONNX Runtime 1.23 on the CPU | x1.17 on the laptop CPU, x0.95-1.03 on the 24-thread one: not worth a second runtime |

Not tried yet, would need the full revalidation: fewer flips at inference (2 instead of 4),
fewer rotations in the filter, reusing the main pass's features in the filter (retraining
it).
Scripts: work_dirs/exp_speed (maps.py; logs maps_dc2.log, polyfilter_lc13cdc2.log,
paired_lc13cdc2.log).

## Caveats

- **Ground truth comes from the union of the detectors' candidates**, plus what the
  reviewer drew. Changes no detector flagged count against every model, so recall is
  biased toward those detectors.
- **r01–r06 candidates came from the 7-square fine-tune**, so the shipped model may be
  scored a little harshly on those squares. The first 7 squares don't have this bias, and
  the fine-tune still wins there.
- **Roads:** only 4 labelled roads in the whole set; the model essentially can't detect
  them.

## Next steps

1. **Test set:** 4–5 pairs from places the model has never seen, ideally in a couple of
   countries, captured the same way (about 4 km view width, max resolution, view
   untouched between dates). Candidates from both the shipped and the new model, so
   neither gets an advantage. This answers: how good is the new model in new places,
   does it beat the shipped one there despite slipping on public benchmarks, and does
   the nudge help?
2. Decide the nudge default (see above).
3. If roads matter: draw the new roads in existing squares (r02 and r03 have many) and
   retrain.
4. If it misses a particular kind of place (dense old towns, desert, coastal villas),
   capture a few squares of that kind.
5. If the new model does worse than the shipped one in new places, keep the shipped model
   as default and offer the fine-tune as an extra option.

## Artifacts

- The 13-square model (`weights/dinov2_vitb14_finetuned.decoder.pth`, the plugin's new
  recommended model) and the `work_dirs/loco7`, `loco13` runs were lost on
  2026-10-20 (see `work_dirs/RECOVERY_NOTES.md`). Retrain with `finetune_dinov2.py` on all
  13 squares: 3000 steps, decoder only, `levir_batch 2`.
- 11 leave-one-out fold models survived in `work_dirs/recovered_checkpoints`.
- The Changen2 Hugging Face cache was deleted, so the ensemble detector is unavailable
  offline.
