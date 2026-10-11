import argparse,os,time,requests
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);a=p.parse_args()
for _ in range(180):
    try:os.kill(a.pid,0)
    except ProcessLookupError:raise RuntimeError('ORA server exited; see output/logs/ora_server.log')
    try:
        r=requests.get('http://127.0.0.1:8008/v1/models',timeout=2)
        if r.ok and any(m['id']=='Video-ORA-4B' for m in r.json()['data']):
            print('ORA ready');break
    except requests.RequestException:pass
    time.sleep(2)
else:raise RuntimeError('ORA startup timeout')
