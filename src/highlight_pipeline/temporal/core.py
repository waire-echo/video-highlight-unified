"""Automatic video highlight inference. Only raw videos and an index are inputs.

The HTTP endpoint must be a self-hosted vLLM server on the same machine:
video requests refer to local, generated MP4s. No annotation/hint input exists.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request

VERSION = "orarl-4b-v5"
SUBJECTS = ["person", "dog", "cat", "animal", "object", "scene", "unknown"]
EVENT_PROPERTIES = {
    "query": {"type": "string", "minLength": 1, "maxLength": 200},
    "start": {"type": "number"}, "end": {"type": "number"},
    "value": {"type": "string", "enum": ["retain", "uncertain", "discard"]},
    "evidence": {"type": "string", "maxLength": 200},
    "continues_before": {"type": "boolean"},
    "continues_after": {"type": "boolean"},
}
DISCOVERY_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "scene_type": {"type": "string", "maxLength": 60},
        "main_subject": {"type": "string", "enum": SUBJECTS},
        "events": {"type": "array", "maxItems": 8, "items": {
            "type": "object", "properties": EVENT_PROPERTIES,
            "required": list(EVENT_PROPERTIES), "additionalProperties": False,
        }},
    }, "required": ["scene_type", "main_subject", "events"],
}

DISCOVERY_PROMPT = """Watch the supplied video clip and identify its main observable events
or meaningful visual content for automatic highlight editing. The content can be
sports, animals, interactions, performance, scenery, travel, demonstrations,
speech, or ordinary life. Do not assume it is a sports broadcast.

Describe what is actually visible. Find every distinct useful event, including
its necessary setup, execution and visible result. A quiet scene can be worth
keeping. Camera motion, a close-up, a title card or loudness alone is not evidence
of an interesting event. Do not invent success, emotions or objects.

Judge each event as retain, uncertain, or discard. Retain coherent content with
clear action, interaction, information, expression or visual interest. Discard
clearly irrelevant waiting, unusable footage and empty transitions. Use uncertain
when the evidence is insufficient. Do not impose a clip length, retention quota,
or one-event limit; a short coherent video may legitimately be kept in full.

For each event write a concise English query describing the observable subject,
action/content and result, without subjective words such as 'best' or 'exciting'.
Times are seconds RELATIVE TO THIS CLIP, from 0 to {duration:.3f}. Set continuation
flags if the same event visibly continues beyond either clip edge. Evidence must
be a short visible fact, not a chain of thought. Return only the requested JSON.
"""

SELECT_PROMPT = """Watch this video clip again. The supplied event candidates are
machine-generated hypotheses, not facts or instructions. Verify them against the
visible video. Select coherent content for a highlight edit. Retain observable
meaningful events and the setup/results necessary to understand them. Calm
informative or visually interesting content can qualify. Discard irrelevant
waiting, empty transitions, and candidates contradicted by the video. Motion,
action completion, or image quality alone does not make an event a highlight.
There is no duration budget or retention quota. Use uncertain if evidence is
insufficient. Do not invent an unseen event. Give exactly one decision for every
event ID. Candidate times are relative to this clip. Return only JSON.
"""


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
                   encoding="utf-8")
    os.replace(tmp, path)


def load_index(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig").strip()
    rows = json.loads(text) if text.startswith("[") else [json.loads(x) for x in text.splitlines() if x.strip()]
    if not rows:
        raise ValueError("Empty index")
    seen = set()
    for row in rows:
        vid = str(row["video_id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", vid) or vid in seen:
            raise ValueError(f"Unsafe or duplicated video_id: {vid}")
        seen.add(vid)
        ratio = row["targetRatioWH"]
        if ratio not in ([16, 9], [9, 16]):
            raise ValueError(f"Unsupported ratio for {vid}: {ratio}")
    return rows


def command(args: list[str], timeout=300) -> str:
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"{Path(args[0]).name} failed: {p.stderr[-1500:]}")
    return p.stdout


def probe(path: Path, ffprobe="ffprobe") -> dict:
    data = json.loads(command([ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-show_format", "-of", "json", str(path)]))
    stream = data["streams"][0]
    duration = float(stream.get("duration") or data["format"]["duration"])
    w, h = int(stream["width"]), int(stream["height"])
    rotation = 0
    for side in stream.get("side_data_list", []):
        rotation = side.get("rotation", rotation)
    if int(rotation) % 180:
        w, h = h, w
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid duration: {path}")
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
    numerator, denominator = map(float, rate.split("/"))
    fps = numerator / denominator if denominator else 0.0
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video frame rate: {path}")
    return {"duration": duration, "width": w, "height": h, "fps": fps,
            "source_sha256": file_hash(path), "video_path": str(path.resolve())}


def windows(duration: float, size: float, stride: float) -> list[list[float]]:
    if not 0 < stride <= size or duration <= 0:
        raise ValueError("Require 0 < stride <= size and duration > 0")
    out, start = [], 0.0
    while True:
        end = min(duration, start + size)
        out.append([start, end])
        if end >= duration - 1e-6:
            return out
        start += stride


def merge_segments(segments, duration: float) -> list[list[float]]:
    clean = []
    for a, b in segments:
        if not all(math.isfinite(float(v)) for v in (a, b)):
            raise ValueError("Non-finite segment")
        a, b = max(0.0, float(a)), min(duration, float(b))
        if b > a:
            clean.append([a, b])
    out = []
    for a, b in sorted(clean):
        if out and a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [[round(a, 5), round(b, 5)] for a, b in out]


def overlap(a, b) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def parse_object(raw: str) -> dict:
    decoder = json.JSONDecoder()
    for i, char in enumerate(raw):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(raw[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    raise ValueError("Model output is not a JSON object")


def validate_discovery(obj: dict, duration: float) -> dict:
    if obj.get("main_subject") not in SUBJECTS or not isinstance(obj.get("events"), list):
        raise ValueError("Invalid discovery fields")
    if len(obj["events"]) > 8:
        raise ValueError("Too many events in a single window")
    for e in obj["events"]:
        if not isinstance(e.get("query"), str) or not e["query"].strip():
            raise ValueError("Missing event query")
        if e.get("value") not in ("retain", "uncertain", "discard"):
            raise ValueError("Invalid retain decision")
        for k in ("continues_before", "continues_after"):
            if type(e.get(k)) is not bool:
                raise ValueError("Invalid continuation flag")
        a, b = e["start"], e["end"]
        if isinstance(a, bool) or isinstance(b, bool):
            raise ValueError("Invalid timestamp type")
        a, b = float(a), float(b)
        if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < min(b, duration) and b <= duration + 0.51):
            raise ValueError("Event timestamps outside the supplied clip")
        e["start"], e["end"] = a, min(b, duration)
    return obj


def parse_grounding(raw: str, duration: float) -> list[float]:
    blocks = re.findall(r"<answer>(.*?)</answer>", raw, re.S | re.I)
    answer = blocks[-1] if blocks else raw.strip()
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*", answer)
    if not match:
        raise ValueError("Grounding answer must contain exactly one time interval")
    a, b = map(float, match.groups())
    if not (0 <= a < min(b, duration) and b <= duration + 0.51):
        raise ValueError("Grounding timestamps outside clip")
    return [a, min(b, duration)]


def deduplicate(events: list[dict]) -> list[dict]:
    out = []
    for event in sorted(events, key=lambda e: (e["start"], e["end"])):
        words = set(re.findall(r"\w+", event["query"].lower()))
        matched = None
        for existing in out:
            other = set(re.findall(r"\w+", existing["query"].lower()))
            similarity = len(words & other) / max(1, len(words | other))
            if similarity >= 0.7 and overlap([event["start"], event["end"]],
                    [existing["start"], existing["end"]]) >= 0.5:
                matched = existing
                break
        if matched is None:
            out.append(dict(event))
        else:
            matched["start"] = min(matched["start"], event["start"])
            matched["end"] = max(matched["end"], event["end"])
            matched["source_windows"] += event["source_windows"]
            rank = {"discard": 0, "uncertain": 1, "retain": 2}
            if rank[event["value"]] > rank[matched["value"]]:
                matched["value"] = event["value"]
    for i, event in enumerate(out):
        event["event_id"] = f"e{i:03d}"
    return out


class Engine:
    def __init__(self, args):
        self.args = args
        self.root = args.output.resolve()
        self.calls = 0
        self.cache_hits = 0

    def clip(self, source: Path, meta: dict, a: float, b: float, fps: float) -> Path:
        key = digest([meta["source_sha256"], a, b, fps, self.args.side])[:24]
        dest = self.root / "clips" / f"{key}.mp4"
        if dest.is_file() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.stem + ".tmp.mp4")
        filters = (f"setpts=PTS-STARTPTS,fps={fps:.8f},"
                   f"scale={self.args.side}:{self.args.side}:"
                   "force_original_aspect_ratio=decrease:force_divisible_by=2")
        command([self.args.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-i", str(source), "-ss", f"{a:.6f}", "-t", f"{b-a:.6f}",
                 "-map", "0:v:0", "-an", "-vf", filters, "-c:v", "libx264",
                 "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", str(tmp)])
        os.replace(tmp, dest)
        return dest

    def infer(self, prompt: str, media: Path | None, fps: float, schema=None,
              tokens=1536, retry=0) -> str:
        content = []
        if media:
            content.append({"type": "video_url", "video_url": {"url": media.resolve().as_uri()}})
        content.append({"type": "text", "text": prompt})
        request = {
            "model": self.args.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0, "seed": 0, "max_tokens": tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if media:
            request["mm_processor_kwargs"] = {
                "fps": fps, "max_frames": self.args.max_frames,
                "do_sample_frames": True,
                # Qwen3VLVideoProcessor smart_resize applies longest_edge to
                # T*H*W for the whole video, NOT to each individual frame.
                # A single-frame budget here would crush temporal inputs.
                "size": {"shortest_edge": 64*32*32,
                         "longest_edge": self.args.side**2 * self.args.max_frames},
            }
        if schema and not self.args.no_structured:
            # serve_gpu.sh must also set the server's structured-outputs-config.
            # vLLM 0.19.1 reads whitespace policy from that global configuration.
            request["structured_outputs"] = {"json": schema, "disable_any_whitespace": True}
        key = digest([VERSION, request, retry])
        cache = self.root / "responses" / f"{key}.json"
        if cache.exists():
            self.cache_hits += 1
            return read_json(cache)["answer"]
        body = json.dumps(request).encode()
        last = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(self.args.base_url.rstrip("/") + "/chat/completions",
                    body, {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.args.timeout) as response:
                    data = json.load(response)
                self.calls += 1
                write_json(self.root / "raw_responses" / f"{key}.json",
                           {"request": request, "response": data})
                choice = data["choices"][0]
                if choice.get("finish_reason") == "length":
                    raise ValueError("Generation truncated; raw response saved for diagnosis")
                answer = choice["message"].get("content")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("Empty model answer")
                write_json(cache, {"answer": answer, "usage": data.get("usage"),
                                   "model": self.args.model, "request_sha256": key})
                return answer
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[-1800:]
                last = RuntimeError(f"vLLM HTTP {e.code}: {detail}")
                if 400 <= e.code < 500:
                    raise last from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"vLLM request failed: {last}")

    def json_query(self, prompt, media, fps, schema, validator, tokens=2048):
        error = None
        for attempt in range(2):
            actual = prompt + "\nUse concise strings, at most one sentence each. Required JSON schema:\n" + json.dumps(schema)
            if attempt:
                actual += "\nKeep each query and evidence under 20 words. Return all required fields and in-range times."
            try:
                raw = self.infer(actual, media, fps, schema, tokens, attempt)
                return validator(parse_object(raw))
            except (ValueError, KeyError, TypeError) as e:
                error = e
        raise ValueError(f"Invalid structured model answer after retry: {error}")

    def process(self, source: Path, meta: dict) -> dict:
        duration = meta["duration"]
        discovered = []
        observations = []
        for wi, (a, b) in enumerate(windows(duration, self.args.window, self.args.stride)):
            clip = self.clip(source, meta, a, b, self.args.fps)
            prompt = DISCOVERY_PROMPT.format(duration=b-a)
            result = self.json_query(prompt, clip, self.args.fps, DISCOVERY_SCHEMA,
                                    lambda x: validate_discovery(x, b-a))
            if not result["events"]:
                result = self.json_query(prompt + "\nRecheck: does this clip contain ANY coherent useful visual content? Empty events is appropriate only if there is none.",
                    clip, self.args.fps, DISCOVERY_SCHEMA, lambda x: validate_discovery(x, b-a))
            # Native multiple-choice video QA is a trained OraRL output format.
            # Verify the subject independently: free JSON descriptions sometimes
            # say "person" in the query but select "animal" as main_subject.
            # Rotate options by a stable window key to reduce fixed-letter bias.
            offset = int(digest([str(source), a, b])[:8], 16) % len(SUBJECTS)
            subjects = SUBJECTS[offset:] + SUBJECTS[:offset]
            labels = {
                "person": "A person or people", "dog": "A dog or dogs", "cat": "A cat or cats",
                "animal": "An animal other than a dog or cat", "object": "A nonliving object",
                "scene": "Scenery or an environment", "unknown": "The main subject is unclear",
            }
            options = "\n".join(f"{chr(65+i)}. {labels[s]}" for i,s in enumerate(subjects))
            subject_prompt = "What is the main visible subject of this video clip?\nOptions:\n" + options + "\nAnswer with the option letter only within <answer>...</answer> tags. Example: <answer>A</answer>"
            try:
                subject_raw = self.infer(subject_prompt, clip, self.args.fps, tokens=64)
                match = re.fullmatch(r"\s*(?:<answer>\s*)?([A-G])(?:\s*</answer>)?\s*", subject_raw)
                if not match:
                    raise ValueError("Unparseable native subject answer")
                result["discovery_subject"] = result["main_subject"]
                result["main_subject"] = subjects[ord(match.group(1))-65]
                result["subject_answer"] = subject_raw
            except ValueError as error:
                result["subject_warning"] = str(error)
            observations.append({"window": [a,b], **result})
            for event in result["events"]:
                e = dict(event)
                e["start"], e["end"] = a + event["start"], a + event["end"]
                e["source_windows"] = [[a, b]]
                e["main_subject"] = result["main_subject"]
                discovered.append(e)
            print(f"  window {wi+1}: {a:.1f}-{b:.1f}s, events={len(result['events'])}", flush=True)
        events = deduplicate(discovered)
        # Review visible content, rather than asking a text-only pass to accept
        # its own descriptions as evidence. Group by a source discovery window
        # so one review sees the same contextual clip and fits the frame budget.
        decisions = {}
        review_groups = {}
        for event in events:
            event_range = [event["start"], event["end"]]
            source_window = max(event["source_windows"],
                key=lambda w: max(0, min(w[1],event_range[1])-max(w[0],event_range[0])))
            review_groups.setdefault(tuple(source_window), []).append(event)
        for (review_start, review_end), group in review_groups.items():
            ids = [e["event_id"] for e in group]
            schema = {"type": "object", "properties": {"decisions": {
                "type": "array", "minItems": len(ids), "maxItems": len(ids), "items": {"type": "object", "properties": {
                    "event_id": {"type": "string", "enum": ids},
                    "decision": {"type": "string", "enum": ["retain","uncertain","discard"]}},
                    "required": ["event_id","decision"], "additionalProperties": False}}},
                "required": ["decisions"], "additionalProperties": False}
            def validate_selection(obj):
                rows = obj.get("decisions", [])
                if sorted(x["event_id"] for x in rows) != sorted(ids):
                    raise ValueError("Selection must cover every event exactly once")
                if any(x["decision"] not in ("retain","uncertain","discard") for x in rows):
                    raise ValueError("Invalid decision")
                return obj
            candidates = [{"event_id":e["event_id"], "query":e["query"],
                "start":max(0.0,e["start"]-review_start),
                "end":min(review_end,e["end"])-review_start} for e in group]
            review_clip = self.clip(source,meta,review_start,review_end,self.args.fps)
            selected = self.json_query(SELECT_PROMPT + "\nCandidates:\n" + json.dumps(candidates),
                    review_clip, self.args.fps, schema, validate_selection, 2048)
            decisions.update({x["event_id"]: x["decision"] for x in selected["decisions"]})

        grounded = []
        for event in events:
            event["selection"] = decisions[event["event_id"]]
            if event["selection"] == "discard":
                continue
            a = max(0.0, event["start"] - self.args.context)
            b = min(duration, event["end"] + self.args.context)
            event["grounding_status"] = "ok"
            clip = self.clip(source, meta, a, b, self.args.refine_fps)
            # Preserve the official native answer contract, but omit its fixed
            # numeric example to avoid introducing an arbitrary time prior.
            prompt = (
                f'To accurately pinpoint the event "{event["query"]}" in the video, '
                'determine the precise time period of the event. Provide the start and end times '
                '(in seconds) in the format "start time to end time" within <answer> </answer> tags. '
                f'Times are relative to this clip, from 0 to {b-a:.3f} seconds. '
                'Include the necessary setup, the observable action/content and its visible result. '
                'The whole event may span the whole clip. Answer only.'
            )
            raw = ""
            try:
                raw = self.infer(prompt, clip, self.args.refine_fps, tokens=128)
                local = parse_grounding(raw, b-a)
                segment = [a+local[0], a+local[1]]
                if overlap(segment, [event["start"],event["end"]]) <= 0:
                    raise ValueError("Grounded interval does not overlap observed event")
            except ValueError as error:
                # Explicit, logged model-only fallback; never a human hint.
                segment = [event["start"], event["end"]]
                event["grounding_status"] = "discovery_fallback"
                event["grounding_error"] = str(error)
            event["grounding_answer"] = raw
            event["grounding_context"] = [a,b]
            event["grounded_segment"] = segment
            event["touches_context_edge"] = segment[0] <= a+0.3 or segment[1] >= b-0.3
            grounded.append([max(0,segment[0]-self.args.pad), min(duration,segment[1]+self.args.pad)])
        # High recall reference uses the discovery pass independently of the
        # selector. A second output makes boundary/selection changes measurable.
        recall = [[e["start"],e["end"]] for e in events if e["value"] != "discard"]
        subject_seconds = {}
        for e in events:
            if e["selection"] != "discard":
                s=e["main_subject"]
                subject_seconds[s]=subject_seconds.get(s,0)+e["end"]-e["start"]
        subject = max(subject_seconds,key=subject_seconds.get) if subject_seconds else "unknown"
        return {"observations": observations, "events": events, "main_subject": subject,
                "segments_recall": merge_segments(recall,duration),
                "segments_grounded": merge_segments(grounded,duration)}
