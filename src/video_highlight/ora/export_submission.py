#!/usr/bin/env python3
"""Export completed model records to segment hints and dense centered JSONL.

Uses original frame timestamps (PTS), including variable-frame-rate videos.
Fails on incomplete records. Centered JSONL is an input to the YOLO backend.
"""
import argparse
from bisect import bisect_left
import json
import math
import os
from pathlib import Path

from core import command, file_hash, load_index, merge_segments, read_json, write_json


def frame_times(path: Path, ffprobe: str) -> list[float]:
    data=json.loads(command([ffprobe,"-v","error","-select_streams","v:0",
        "-show_frames","-show_entries","frame=best_effort_timestamp_time",
        "-of","json",str(path)]))
    frames=data.get("frames",[])
    if not frames or any("best_effort_timestamp_time" not in x for x in frames):
        raise ValueError(f"Missing original frame timestamps: {path}")
    values=[float(x["best_effort_timestamp_time"]) for x in frames]
    if any(not math.isfinite(x) for x in values) or any(b<a for a,b in zip(values,values[1:])):
        raise ValueError(f"Invalid/nonmonotonic frame timestamps: {path}")
    return [x-values[0] for x in values]


def selected_frames(times: list[float], segments) -> list[int]:
    result=set()
    for a,b in segments:
        first=bisect_left(times,a-1e-8)
        last=bisect_left(times,b-1e-8)
        result.update(range(first,last))
    return sorted(result)


def inclusive_ranges(frames: list[int]) -> list[list[int]]:
    ranges = []
    for i in frames:
        if ranges and i == ranges[-1][1] + 1:
            ranges[-1][1] = i
        else:
            ranges.append([i,i])
    return ranges


def center_box(width: int, height: int, ratio: list[int]) -> list[int]:
    tw,th=ratio
    w=min(width,math.floor(height*tw/th))
    h=w*th/tw
    if w<=0:
        raise ValueError("Video is too small for requested crop")
    return [math.floor((width-w)/2),math.floor((height-h)/2),w]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w",encoding="utf-8",newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=Path,required=True)
    p.add_argument("--videos",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--variant",choices=["recall","grounded"],default="grounded")
    p.add_argument("--ffprobe",default="ffprobe")
    p.add_argument("--temporal-only",action="store_true",help="Write only time intervals and frame IDs; no crop boxes or YOLO hints")
    args=p.parse_args()
    index=load_index(args.index)
    manifest=read_json(args.run/"manifest.json")
    if manifest["input_ids"]!=[str(r["video_id"]) for r in index] or manifest["index_sha256"]!=file_hash(args.index):
        raise ValueError("Index does not match this inference run")
    records=[]
    for item in index:
        path=args.run/"records"/f"{item['video_id']}.json"
        if not path.exists():
            raise ValueError(f"Incomplete run; missing record {item['video_id']}")
        row=read_json(path)
        if row.get("status")!="complete" or row.get("signature")!=manifest["signature"]:
            raise ValueError(f"Incomplete/stale record {item['video_id']}")
        if row["targetRatioWH"]!=item["targetRatioWH"]:
            raise ValueError("Ratio mismatch")
        source=args.videos/f"{item['video_id']}.mp4"
        if file_hash(source)!=row["source_sha256"]:
            raise ValueError(f"Source video changed: {source}")
        records.append(row)
    out=args.run/f"export_{args.variant}"
    hints=out/"hints"
    rows=[]
    dense_frames=[]
    handoff=[]
    reports=[]
    for record in records:
        vid=str(record["video_id"])
        cache=args.run/"timestamps"/f"{vid}.json"
        cached=read_json(cache) if cache.exists() else {}
        if cached.get("source_sha256")==record["source_sha256"]:
            times=cached["times"]
        else:
            times=frame_times(args.videos/f"{vid}.mp4",args.ffprobe)
            write_json(cache,{"source_sha256":record["source_sha256"],"times":times})
        segments=merge_segments(record[f"segments_{args.variant}"],record["duration"])
        frames=selected_frames(times,segments)
        frame_ranges=inclusive_ranges(frames)
        if any(type(i) is not int or not 0<=i<len(times) for i in frames):
            raise ValueError(f"Invalid selected frame in {vid}")
        dense_frames.append({"video_id":record["video_id"],"targetRatioWH":record["targetRatioWH"],
            "frame_count":len(times),"frame_index_base":0,"highlight_frames":frames})
        # Paired second intervals represent the exact selected frame runs.
        # Preserve the model's continuous intervals separately for diagnostics.
        paired_seconds=[[times[a],times[b+1] if b+1<len(times) else record["duration"]]
                        for a,b in frame_ranges]
        queries=[e["query"] for e in record["events"] if
                 (e["value"] if args.variant=="recall" else e.get("selection",e["value"]))!="discard"]
        fallback_count=sum(e.get("grounding_status")=="discovery_fallback" for e in record["events"])
        handoff.append({"schema_version":"temporal-handoff/v1", "video_id":record["video_id"],
            "targetRatioWH":record["targetRatioWH"], "fps":record.get("fps",len(times)/record["duration"]),
            "frame_count":len(times), "duration_sec":record["duration"],
            "segments_sec":paired_seconds, "segments_frame":frame_ranges,
            "event_segments_sec":segments, "timestamp_mapping":"original_frame_pts",
            "frame_index_convention":"inclusive", "subject_hint":record["main_subject"],
            "action_description":"; ".join(queries), "source":f"video_ora_4b_{args.variant}",
            "valid":True, "fallback_used":"automatic_discovery" if args.variant=="grounded" and fallback_count else None,
            "errors":[], "warnings":["No event retained"] if not frames else [],
            "raw_diagnostics_path":str((args.run/"records"/f"{vid}.json").resolve())})
        if not args.temporal_only:
            box=center_box(record["width"],record["height"],record["targetRatioWH"])
            rows.append({"video_id":record["video_id"],"targetRatioWH":record["targetRatioWH"],
                         "predictions":[{"frame":i,"bboxes":box.copy()} for i in frames]})
            write_json(hints/f"{vid}.json",{
            "video_id":record["video_id"],"segments":segments,
            "main_subject":record["main_subject"],
            # Existing YOLO backend expects a categorical subject hint.
            "subject_hint":record["main_subject"],
            "event_queries":queries,
            "origin":"raw_video_model_inference", "variant":args.variant,
            "source_sha256":record["source_sha256"],
        })
        reports.append({"video_id":vid,"source_frames":len(times),"selected_frames":len(frames),
            "coverage":len(frames)/len(times),"segments":segments,
            "grounding_fallbacks":sum(e.get("grounding_status")=="discovery_fallback" for e in record["events"])})
    if not args.temporal_only:
        write_jsonl(out/"center.jsonl",rows)
    write_jsonl(out/"highlight_frames.jsonl",dense_frames)
    write_jsonl(out/"temporal_handoff.jsonl",handoff)
    write_json(out/"report.json",{"variant":args.variant,"videos":len(handoff),
        "prediction_frames":sum(len(r["highlight_frames"]) for r in dense_frames),
        "empty_videos":sum(not r["highlight_frames"] for r in dense_frames),
        "note":"Coverage and fallback counts are diagnostics, not accuracy measurements.","details":reports})
    print(f"Exported {len(handoff)} videos to {out}; temporal_only={args.temporal_only}",flush=True)


if __name__=="__main__":
    main()
