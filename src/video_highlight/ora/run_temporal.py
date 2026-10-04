#!/usr/bin/env python3
"""Run fully automatic discovery, selection and temporal grounding on vLLM."""
import argparse
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlparse

from core import Engine, VERSION, digest, file_hash, load_index, probe, read_json, windows, write_json
from video_highlight.prompts import bundle_hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=Path,required=True)
    p.add_argument("--videos",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--base-url",default="http://127.0.0.1:8000/v1")
    p.add_argument("--model",default="Video-ORA-4B")
    p.add_argument("--model-revision",default="",help="Snapshot commit; normally read from TASK_ROOT/model_provenance.json")
    p.add_argument("--window",type=float,default=32)
    p.add_argument("--stride",type=float,default=24)
    p.add_argument("--fps",type=float,default=2)
    p.add_argument("--refine-fps",type=float,default=4)
    p.add_argument("--side",type=int,default=448)
    p.add_argument("--max-frames",type=int,default=128)
    p.add_argument("--context",type=float,default=3)
    p.add_argument("--pad",type=float,default=0.25)
    p.add_argument("--timeout",type=float,default=300)
    p.add_argument("--ffmpeg",default="ffmpeg")
    p.add_argument("--ffprobe",default="ffprobe")
    p.add_argument("--smoke",type=int,default=0,help="Select N videos by duration quantiles; no submission is made")
    p.add_argument("--plan",action="store_true",help="Probe input and estimate number of windows without a model")
    p.add_argument("--no-structured",action="store_true",help="Diagnostic fallback for servers without JSON schema support")
    args=p.parse_args()
    if urlparse(args.base_url).hostname not in ("127.0.0.1","localhost","::1"):
        p.error("Run this client beside the self-hosted server; base-url must be loopback")
    if not 0 < args.stride <= args.window or min(args.fps,args.refine_fps,args.side,args.max_frames,args.timeout)<=0:
        p.error("Invalid sampling/window parameters")
    if min(args.context,args.pad,args.smoke)<0:
        p.error("Context, padding and smoke count must be nonnegative")
    rows=load_index(args.index)
    provenance=args.output.resolve().parent/"model_provenance.json"
    if not args.model_revision and provenance.exists():
        args.model_revision=read_json(provenance)["revision"]
    cfg={k:v for k,v in vars(args).items() if k not in ("index","videos","output","smoke","plan","ffmpeg","ffprobe","base_url","timeout")}
    cfg['prompt_hashes'] = bundle_hashes({'ora/v5': 'ora/v5'})
    signature=digest([VERSION,cfg,file_hash(Path(__file__).with_name("core.py"))])
    manifest_path=args.output/"manifest.json"
    manifest={"version":VERSION,"signature":signature,"configuration":cfg,
              "index_sha256":file_hash(args.index),"input_ids":[str(x["video_id"]) for x in rows]}
    if manifest_path.exists() and read_json(manifest_path)!=manifest:
        raise RuntimeError("Output belongs to another configuration/index; choose a new --output directory")
    write_json(manifest_path,manifest)
    metadata={}
    for row in rows:
        vid=str(row["video_id"])
        path=args.videos/f"{vid}.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata[vid]=probe(path,args.ffprobe)
    total_windows=sum(len(windows(m["duration"],args.window,args.stride)) for m in metadata.values())
    plan={"videos":len(rows),"total_minutes":sum(x["duration"] for x in metadata.values())/60,
          "discovery_windows":total_windows,"note":"Extra calls: visual review per candidate-bearing discovery window, grounding per retained event; empty discovery is rechecked."}
    write_json(args.output/"plan.json",plan)
    print(json.dumps(plan,ensure_ascii=False),flush=True)
    if args.plan:
        return 0
    if args.smoke:
        ordered=sorted(rows,key=lambda r:metadata[str(r["video_id"])]["duration"])
        n=min(args.smoke,len(rows))
        indices=sorted({round(i*(len(rows)-1)/max(1,n-1)) for i in range(n)})
        rows=[ordered[i] for i in indices]
    engine=Engine(args)
    failures=[]
    begin=time.monotonic()
    processed=0
    for position,row in enumerate(rows,1):
        vid=str(row["video_id"])
        meta=metadata[vid]
        record_path=args.output/"records"/f"{vid}.json"
        if record_path.exists():
            old=read_json(record_path)
            if old.get("status")=="complete" and old.get("signature")==signature and old.get("source_sha256")==meta["source_sha256"]:
                print(f"[{position}/{len(rows)}] {vid}: cached",flush=True)
                continue
        print(f"[{position}/{len(rows)}] {vid}: {meta['duration']:.2f}s",flush=True)
        started=time.monotonic()
        record={"video_id":row["video_id"],"targetRatioWH":row["targetRatioWH"],
                "signature":signature,**meta}
        try:
            record.update(engine.process(args.videos/f"{vid}.mp4",meta))
            record["status"]="complete"
        except Exception as error:
            record.update(status="error",error=f"{type(error).__name__}: {error}")
            failures.append(vid)
        record["elapsed_seconds"]=round(time.monotonic()-started,3)
        write_json(record_path,record)
        processed+=1
        elapsed=time.monotonic()-begin
        print(f"  {record['status']} {record['elapsed_seconds']}s; calls={engine.calls}; cache_hits={engine.cache_hits}",flush=True)
        if record["status"]=="error":
            print(record["error"],flush=True)
            if len(failures)>=3 and processed==len(failures):
                print("First three videos failed: stopping for environment repair.",flush=True)
                break
    report={"requested_videos":len(rows),"processed_this_run":processed,
            "failed_ids":failures,"model_calls":engine.calls,"cached_calls":engine.cache_hits,
            "elapsed_seconds":round(time.monotonic()-begin,2),"smoke":bool(args.smoke)}
    write_json(args.output/("smoke_report.json" if args.smoke else "run_report.json"),report)
    print(json.dumps(report),flush=True)
    return 1 if failures else 0


if __name__=="__main__":
    sys.exit(main())
