# editing

Status: `tool`.

Inspection/rendering of selected frames; source videos stay outside Git.

Shared implementation: `src/video_highlight/`.

`tools/preview_submission.py` renders selected-frame highlights. `tools/preview_timeline.py` renders the full source timeline with synchronized source/box/crop panels, or compares two submissions. Input/output paths and video IDs are CLI parameters; no experiment-specific source paths or leaderboard labels are hardcoded. See `--help` and `docs/workspace_cleanup.md`.
