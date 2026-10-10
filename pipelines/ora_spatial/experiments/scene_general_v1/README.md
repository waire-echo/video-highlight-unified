# Uniform scenery candidate experiment

This experiment accepts an arbitrary list of input video IDs. IDs are used for lookup and output names only. No category, target, crop geometry, parameter, or prompt is selected by video ID. Previously manually refined scene anchors are not loaded.

For each original selected contiguous segment, split at existing TransNet boundaries:

1. Sample up to five chronological original frames; decode nearby earlier frames for motion.
2. At up to three uniformly spaced samples ask ORA the same generic localization question. These boxes are proposals, not ground truth.
3. Generate constant-scale candidates at 50% and 75% of the largest legal width from ORA centers, spectral saliency, and camera-compensated flow combined with saliency. Include the widest legal centered context candidate. Penalize discontinuous saliency-window paths; apply the same static/panning threshold and pan rate limit to every candidate.
4. Show all candidates at identical times beside original context. ORA selects with the same multiple-choice prompt. Repeat with reversed option order; disagreements receive a pairwise comparison. Parsing failures are recorded, not treated as accurate detections. Unresolved selections keep the widest context.
5. Interpolate the selected bounded camera path onto exactly the original selected frame IDs. Scale is constant per segment. No target-specific containment union and no new SAM propagation are used.

Configuration and source/input hashes are frozen in the output before inference. Existing results are immutable comparison inputs. Only complete caches with the same signature are reusable. Changing the algorithm after seeing individual outputs requires a separately named experiment; do not patch crops manually.

The requested range includes an empty temporal prediction. It remains empty because this experiment changes only spatial crops. It does not invent temporal coverage.

Limitations: sparse samples can miss short events; camera compensation is approximate; ORA may choose semantically wrong crops or prefer full context; saliency can emphasize graphics or highlights. Agreement under reordered choices is consistency evidence, not calibrated correctness. Parameters are initial uniform engineering choices, not official-score-optimized values. Model total remains within the existing parameter ledger; no additional weights are introduced.
