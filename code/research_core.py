# Research functions preserved from paper_final/code/final_table.py.

import os
from math import exp, lgamma, log

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

IOU_MATCH, IOU_DUP = 0.5, 0.5

def _bcf(a,b,x):
    E,F=3e-16,1e-300;qab,qap,qam=a+b,a+1,a-1;c,d=1.,1-qab*x/qap
    d=1/(F if abs(d)<F else d);h=d
    for m in range(1,201):
        m2=2*m;aa=m*(b-m)*x/((qam+m2)*(a+m2))
        d=1+aa*d;d=F if abs(d)<F else d;c=1+aa/c;c=F if abs(c)<F else c
        d=1/d;h*=d*c;aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2))
        d=1+aa*d;d=F if abs(d)<F else d;c=1+aa/c;c=F if abs(c)<F else c
        d=1/d;de=d*c;h*=de
        if abs(de-1)<E:break
    return h

def _bi(a,b,x):
    if x<=0:return 0.
    if x>=1:return 1.
    bt=exp(lgamma(a+b)-lgamma(a)-lgamma(b)+a*log(x)+b*log(1-x))
    return bt*_bcf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-bt*_bcf(b,a,1-x)/b

def _p(p,a,b):
    lo,hi=0.,1.
    for _ in range(200):
        m=(lo+hi)/2
        if _bi(a,b,m)<p:lo=m
        else:hi=m
    return (lo+hi)/2

def ci(k,n): return (0. if k==0 else _p(.025,k,n-k+1), 1. if k==n else _p(.975,k+1,n-k))

def load(p, N, conf=False):
    o=[]
    if not os.path.exists(p): return o
    for l in open(p,encoding='utf-8'):
        q=l.split()
        if len(q)<9: continue
        o.append((N[int(float(q[0]))], Polygon(np.array(list(map(float,q[1:9]))).reshape(4,2)),
                  float(q[-1]) if conf and len(q)>=10 else None))
    return o

def iou(a,b):
    u=a.union(b).area; return a.intersection(b).area/u if u else 0.

def exact(gt, pr):
    used=set()
    for gc,gp,_ in gt:
        b,bi=0,-1
        for i,(pc,pp,_) in enumerate(pr):
            if i in used: continue
            v=iou(gp,pp)
            if v>b: b,bi=v,i
        if bi<0 or b<IOU_MATCH or pr[bi][0]!=gc: return False
        used.add(bi)
    return len(used)==len(pr)

def tp_fp(gt, pr):
    used=set(); out=[]
    for pc,pp,cf in sorted(pr,key=lambda x:-x[2]):
        b,bi=0,-1
        for i,(gc,gp,_) in enumerate(gt):
            if i in used: continue
            v=iou(pp,gp)
            if v>b: b,bi=v,i
        ok = bi>=0 and b>=IOU_MATCH and gt[bi][0]==pc
        if ok: used.add(bi)
        out.append((pc,cf,ok))
    return out

def derive_g(gt_dir, pl_dir, N):
    """val에서 클래스별 임계값 유도: TP최소 > FP최대 → 중간값"""
    tp,fp={}, {}
    for f in sorted(os.listdir(gt_dir)):
        if not f.endswith('.txt'): continue
        for c,cf,ok in tp_fp(load(f'{gt_dir}/{f}',N), load(f'{pl_dir}/{f}',N,True)):
            (tp if ok else fp).setdefault(c,[]).append(cf)
    thr={}
    for c in fp:
        if c in tp and min(tp[c])>max(fp[c]): thr[c]=round((min(tp[c])+max(fp[c]))/2,2)
    return thr

def apply_g(pr,thr):  return [x for x in pr if x[2]>=thr.get(x[0],0.)]

def apply_g2(pr):
    keep=[]
    for pc,pp,cf in sorted(pr,key=lambda x:-x[2]):
        if not any(iou(pp,kp)>=IOU_DUP for _,kp,_ in keep): keep.append((pc,pp,cf))
    return keep

def score(gt_dir, pl_dir, N, post=None):
    k=n=0
    for f in sorted(os.listdir(gt_dir)):
        if not f.endswith('.txt'): continue
        pr=load(f'{pl_dir}/{f}',N,True)
        if post: pr=post(pr)
        k+=exact(load(f'{gt_dir}/{f}',N),pr); n+=1
    return k,n
