"""Non-neural spectral saliency + median-camera-compensated optical flow.

This is an uncertainty fallback, not a semantic detector. Diffuse maps return
the image center. Scores below are concentration measures, not probabilities.
"""
import cv2
import numpy as np


def fallback_focus(bgr,previous=None):
    small=cv2.resize(bgr,(160,90))
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY).astype(np.float32)
    spectrum=np.fft.fft2(gray+1)
    logamp=np.log(np.abs(spectrum)+1e-8)
    residual=logamp-cv2.blur(logamp,(3,3))
    saliency=np.abs(np.fft.ifft2(np.exp(residual+1j*np.angle(spectrum))))**2
    saliency=cv2.GaussianBlur(saliency.astype(np.float32),(9,9),2)
    saliency/=saliency.sum()+1e-9
    if previous is not None:
        old=cv2.cvtColor(cv2.resize(previous,(160,90)),cv2.COLOR_BGR2GRAY)
        flow=cv2.calcOpticalFlowFarneback(old,gray.astype(np.uint8),None,0.5,3,15,3,5,1.2,0)
        # Approximate global translation; rotation/zoom still require conservative fallback.
        global_flow=np.median(flow.reshape(-1,2),axis=0)
        motion=np.linalg.norm(flow-global_flow,axis=2)
        motion=np.maximum(0,motion-np.percentile(motion,70))
        if motion.sum()>1e-5:
            saliency=0.8*saliency+0.2*motion/motion.sum()
    cells=cv2.resize(saliency,(8,5),interpolation=cv2.INTER_AREA)
    concentration=float(cells.max()/(cells.sum()+1e-12))
    if concentration<0.12:
        return [bgr.shape[1]/2,bgr.shape[0]/2],0.0
    mask=np.maximum(saliency-np.percentile(saliency,90),0)
    yy,xx=np.indices(mask.shape)
    mass=mask.sum()
    if mass<=1e-9:return [bgr.shape[1]/2,bgr.shape[0]/2],0.0
    point=[float((xx*mask).sum()/mass/160*bgr.shape[1]),float((yy*mask).sum()/mass/90*bgr.shape[0])]
    return point,min(0.7,max(0,(concentration-0.12)/0.18))
