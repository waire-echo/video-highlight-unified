"""Read-only five-timepoint inspection from recorded predictions and original frames."""
import argparse
from temporal import *

p=argparse.ArgumentParser();p.add_argument('--record',required=True);p.add_argument('--video',required=True);p.add_argument('--output',required=True);a=p.parse_args()
rec=read_json(a.record);samples=rec['sample_frames'];chosen=sorted(set(np.linspace(0,len(samples)-1,min(5,len(samples))).round().astype(int)))
frames=read_frames(a.video,[samples[i] for i in chosen],960)
cap=cv2.VideoCapture(a.video);W=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));H=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
tilew=640;tileh=round(tilew*H/W);sheet=np.zeros((len(chosen)*(tileh+38),2*tilew,3),np.uint8)
for r,i in enumerate(chosen):
    im=frames[samples[i]];obs=rec['observations'][i];target=rec['targets'][i];x,y,w=rec['sample_boxes'][i];h=w*9/16
    sx,sy=im.shape[1]/W,im.shape[0]/H;shown=marked(im,target,(W,H));cv2.rectangle(shown,(round(x*sx),round(y*sy)),(round((x+w)*sx),round((y+h)*sy)),(0,180,255),2)
    crop=im[int(y*sy):max(int((y+h)*sy),int(y*sy)+1),int(x*sx):max(int((x+w)*sx),int(x*sx)+1)]
    top=r*(tileh+38);cv2.putText(sheet,f"frame {samples[i]} | {obs['state']} | {obs.get('verdict')} | width {w}",(6,top+25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
    sheet[top+38:top+38+tileh,:tilew]=cv2.resize(shown,(tilew,tileh));sheet[top+38:top+38+tileh,tilew:]=cv2.resize(crop,(tilew,tileh))
Path(a.output).parent.mkdir(parents=True,exist_ok=True);cv2.imwrite(a.output,sheet)
