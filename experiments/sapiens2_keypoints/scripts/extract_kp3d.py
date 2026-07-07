#!/usr/bin/env python3
"""Extract Sapiens2 3D keypoint positions (via pointmap lookup) for the study manifest."""
import sys, os, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import safetensors.torch as sf
import importlib.util
from PIL import Image

_spec=importlib.util.spec_from_file_location("s2","/tmp/sapiens2/sapiens/backbones/standalone/sapiens2.py")
_s2=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s2); Sapiens2=_s2.Sapiens2
H,W=1024,768
CK_PM="/tmp/sapiens2_checkpoints/pointmap/sapiens2_1b_pointmap.safetensors"
CK_POSE="/tmp/sapiens2_checkpoints/pose/sapiens2_1b_pose.safetensors"

class DenseHead(nn.Module):
    def __init__(self,embed_dim=1536,up_co4=(6144,3072,2048,1024)):
        super().__init__()
        self.input_conv=nn.Sequential(nn.Conv2d(embed_dim,embed_dim,3,padding=1),nn.InstanceNorm2d(embed_dim),nn.SiLU(True))
        blocks=[];ci=embed_dim
        for co4 in up_co4:
            blocks.append(nn.Sequential(nn.Conv2d(ci,co4,3,padding=1),nn.PixelShuffle(2),nn.InstanceNorm2d(co4//4),nn.SiLU(True)));ci=co4//4
        self.upsample_blocks=nn.Sequential(*blocks)
        self.conv_layers=nn.Sequential(nn.Conv2d(ci,64,3,padding=1),nn.InstanceNorm2d(64),nn.SiLU(True),
            nn.Conv2d(64,32,3,padding=1),nn.InstanceNorm2d(32),nn.SiLU(True),
            nn.Conv2d(32,16,3,padding=1),nn.InstanceNorm2d(16),nn.SiLU(True))
        self.conv_pointmap=nn.Conv2d(16,3,1)
    def forward(self,fm): return self.conv_pointmap(self.conv_layers(self.upsample_blocks(self.input_conv(fm))))

class PoseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv_layers=nn.Sequential(nn.ConvTranspose2d(1536,1536,4,2,1,bias=False),nn.InstanceNorm2d(1536),nn.SiLU(True),
            nn.ConvTranspose2d(1536,1024,4,2,1,bias=False),nn.InstanceNorm2d(1024),nn.SiLU(True))
        self.conv_layers=nn.Sequential(nn.Conv2d(1024,768,1),nn.InstanceNorm2d(768),nn.SiLU(True),
            nn.Conv2d(768,512,1),nn.InstanceNorm2d(512),nn.SiLU(True),nn.Conv2d(512,256,1),nn.InstanceNorm2d(256),nn.SiLU(True))
        self.conv_pose=nn.Conv2d(256,308,1)
    def forward(self,fm): return self.conv_pose(self.conv_layers(self.deconv_layers(fm)))

def load(ckpt,head,dev):
    bb=Sapiens2(arch='sapiens2_1b',out_type='featmap');ck=sf.load_file(ckpt)
    bb.load_state_dict({k[9:]:v for k,v in ck.items() if k.startswith('backbone.')},strict=False)
    head.load_state_dict({k[12:]:v for k,v in ck.items() if k.startswith('decode_head.')},strict=False)
    return bb.to(dev).eval().half(), head.to(dev).eval().half()

def load_img(path,dev):
    img=Image.open(path).convert('RGB').resize((W,H),Image.LANCZOS)
    arr=(np.array(img).astype(np.float32)/255.0-0.5)/0.5
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(dev).half()

def decode_hm(hm):
    hm=hm[0].float();K,Hh,Wh=hm.shape;flat=hm.reshape(K,-1);conf,idx=flat.max(1)
    ys=(idx//Wh).float();xs=(idx%Wh).float()
    return torch.stack([xs*(W/Wh),ys*(H/Hh),conf],1).cpu().numpy()

def main():
    dev='cuda'
    man=json.load(open('/tmp/eidolon_3d_spike/study_manifest.json'))
    print(f"Extracting 3D keypoints for {len(man)} images...")
    pm_bb,pm_hd=load(CK_PM,DenseHead(),dev)
    po_bb,po_hd=load(CK_POSE,PoseHead(),dev)
    kp3d=np.full((len(man),308,3),np.nan,dtype=np.float32)
    for i,m in enumerate(man):
        t=load_img(m['img'],dev)
        with torch.no_grad():
            pm=pm_hd(pm_bb(t)[0])[0].float().permute(1,2,0).cpu().numpy()
            kp=decode_hm(po_hd(po_bb(t)[0]))
        for k in range(308):
            x,y,c=kp[k];xi,yi=int(round(x)),int(round(y))
            if 0<=xi<W and 0<=yi<H and c>=0.3 and pm[yi,xi,2]>1e-6:
                kp3d[i,k]=pm[yi,xi]
        if (i+1)%50==0: print(f"  {i+1}/{len(man)}")
    np.savez('/tmp/eidolon_3d_spike/study_kp3d.npz', kp3d=kp3d)
    valid=np.isfinite(kp3d).all(2)
    print(f"Saved study_kp3d.npz. Mean valid 3D kp/img: {valid.sum(1).mean():.0f}/308")

if __name__=='__main__': main()
