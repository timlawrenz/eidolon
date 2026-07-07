#!/usr/bin/env python3
"""Extract Sapiens2 308 keypoints (x,y,conf) for the study manifest. Also gathers DWPose + AuraFace."""
import sys, os, json, glob
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import safetensors.torch as sf
import importlib.util
from PIL import Image

_spec=importlib.util.spec_from_file_location("s2","/tmp/sapiens2/sapiens/backbones/standalone/sapiens2.py")
_s2=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s2)
Sapiens2=_s2.Sapiens2
H,W=1024,768
CK_POSE="/tmp/sapiens2_checkpoints/pose/sapiens2_1b_pose.safetensors"

class PoseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv_layers=nn.Sequential(
            nn.ConvTranspose2d(1536,1536,4,2,1,bias=False),nn.InstanceNorm2d(1536),nn.SiLU(True),
            nn.ConvTranspose2d(1536,1024,4,2,1,bias=False),nn.InstanceNorm2d(1024),nn.SiLU(True))
        self.conv_layers=nn.Sequential(
            nn.Conv2d(1024,768,1),nn.InstanceNorm2d(768),nn.SiLU(True),
            nn.Conv2d(768,512,1),nn.InstanceNorm2d(512),nn.SiLU(True),
            nn.Conv2d(512,256,1),nn.InstanceNorm2d(256),nn.SiLU(True))
        self.conv_pose=nn.Conv2d(256,308,1)
    def forward(self,fm): return self.conv_pose(self.conv_layers(self.deconv_layers(fm)))

def load(ckpt,head,dev):
    bb=Sapiens2(arch='sapiens2_1b',out_type='featmap')
    ck=sf.load_file(ckpt)
    bb.load_state_dict({k[9:]:v for k,v in ck.items() if k.startswith('backbone.')},strict=False)
    head.load_state_dict({k[12:]:v for k,v in ck.items() if k.startswith('decode_head.')},strict=False)
    return bb.to(dev).eval().half(), head.to(dev).eval().half()

def load_img(path,dev):
    img=Image.open(path).convert('RGB').resize((W,H),Image.LANCZOS)
    arr=(np.array(img).astype(np.float32)/255.0-0.5)/0.5
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(dev).half()

def decode_hm(hm):
    hm=hm[0].float();K,Hh,Wh=hm.shape
    flat=hm.reshape(K,-1);conf,idx=flat.max(1)
    ys=(idx//Wh).float();xs=(idx%Wh).float()
    return torch.stack([xs*(W/Wh),ys*(H/Hh),conf],1).cpu().numpy()

def main():
    dev='cuda'
    man=json.load(open('/tmp/eidolon_3d_spike/study_manifest.json'))
    print(f"Extracting Sapiens2 keypoints for {len(man)} images...")
    bb,hd=load(CK_POSE,PoseHead(),dev)

    s2kp=np.zeros((len(man),308,3),dtype=np.float32)
    dwkp=np.full((len(man),68,3),np.nan,dtype=np.float32)
    af=np.full((len(man),512),np.nan,dtype=np.float32)
    personas=[]; setids=[]
    for i,m in enumerate(man):
        t=load_img(m['img'],dev)
        with torch.no_grad():
            s2kp[i]=decode_hm(hd(bb(t)[0]))
        # DWPose
        try:
            p=np.load(m['pose'])
            if p.shape[0]>=68: dwkp[i]=p[:68] if p.shape[1]==3 else np.c_[p[:68,:2],np.ones(68)]
        except Exception: pass
        # AuraFace
        try:
            a=np.load(m['af_npy']).ravel()
            if a.shape[0]>=512: af[i]=a[:512]
        except Exception: pass
        personas.append(m['persona']); setids.append(m['set_id'])
        if (i+1)%50==0: print(f"  {i+1}/{len(man)}")

    np.savez('/tmp/eidolon_3d_spike/study_features.npz',
             s2kp=s2kp, dwkp=dwkp, af=af,
             personas=np.array(personas), setids=np.array(setids),
             zg=np.array([m['zg'] for m in man]), af_dist=np.array([m['af'] for m in man]))
    # Data quality report
    dw_ok=np.isfinite(dwkp).all((1,2)).sum()
    af_ok=np.isfinite(af).all(1).sum()
    print(f"\nSaved study_features.npz")
    print(f"  Sapiens2 kp: {len(man)} images x 308 kp")
    print(f"  DWPose valid: {dw_ok}/{len(man)}")
    print(f"  AuraFace valid: {af_ok}/{len(man)}")
    print(f"  Personas: {len(set(personas))}")
    # Sapiens2 confidence stats
    conf=s2kp[:,:,2]
    print(f"  Sapiens2 conf: mean={conf.mean():.3f}, per-image kp>0.3: {(conf>0.3).sum(1).mean():.0f}/308")

if __name__=='__main__': main()
