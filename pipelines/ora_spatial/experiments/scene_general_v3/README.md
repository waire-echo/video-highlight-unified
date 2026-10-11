# Temporal candidate reframing v3

This experiment uses the same rules for every video. IDs are used only to select
inputs and name output files. The baseline remains `inputs/baseline.zip`, the
user-designated YOLO-only submission. There are no manually chosen target words,
coordinates, or per-video thresholds.

## Changes from v2

- New ORA localization every second on each original selected shot/continuous span.
- Reference-conditioned semantic checks against the established and recent focus.
  Position displacement and changing visible area do not veto semantic continuity.
  Failed checks trigger reference-conditioned re-localization and a second check.
- Once established, a target is not replaced with another object just because it
  leaves view. A new shot can establish a new focus.
- Explicit visible/partial/occluded/out_of_view/uncertain states. Short uncertainty
  can retain a target for at most two seconds with reliable camera compensation.
  Exit is not extrapolated; unresolved intervals widen locally.
- Candidate decisions use overlapping windows of four sample intervals. All actual
  panels, semantic responses, candidate coordinates and exclusions are retained.
- Two candidate orders are cyclic derangements: every candidate changes letter.
  Disagreement falls back to the eligible candidate with lowest general cost.
- Variable width, bounded center smoothing and a bounded log-zoom envelope. Visible
  targets constrain smoothing; narrow fixed crops cannot discard valid moving targets.
- Complete source hashes, progress/status, partial validated JSONL, final JSONL/ZIP,
  report and an archive with hashes. New output directories never read past inference.

The model budget is unchanged (conservative inherited ledger: 5,507,029,690).
This candidate-based experiment does not add SAM or Pose inference. Existing shot
boundaries and source/timestamp metadata are retained; ORA and candidate decisions
are recomputed for all requested videos, even videos processed in older experiments.
Original temporal selections, empty entries, target ratios and video order remain.

## Limits

Semantic checks are model estimates, not labels. A wrong seed or repeated semantic
mistake can still create a bad track. Evidence is sampled each second and final
frame crops are interpolated; this is not dense segmentation. Coverage refers to
estimated regions. Impossible aspect-ratio coverage is explicitly reported, never
silently called successful. Visual or official score improvement is not guaranteed.

## Commands

Run `env/bin/python -m unittest discover -s experiments/scene_general_v3 -p
test_temporal.py -v`, start `serve_ora.sh` if the dedicated endpoint is absent,
then run `preflight.py --ids <comma-separated IDs> --video-dir <original videos>`.
Only after preflight succeeds, start `temporal.py` with those arguments and a new
`--output` directory. The output's `run.lock` prevents duplicate jobs. A failed run
can resume only with identical frozen sources; never mutate sources of an active run.

No recurring monitor, contest submission, or instance shutdown is created.
