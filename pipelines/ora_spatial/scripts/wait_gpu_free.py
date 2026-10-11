import subprocess,time
for _ in range(60):
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],text=True)
    if not text.strip():break
    time.sleep(2)
else:raise RuntimeError('GPU still in use after stopping our ORA server; inspect nvidia-smi')
