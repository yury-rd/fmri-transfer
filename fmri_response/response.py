"""Shared-coordinate response prediction; node order BASE,A,F,AF,FA."""
import numpy as np

def fields(a):
 h=np.stack([a[:,:2,k]-a[:,:2,0] for k in (1,2,3)],axis=-1)
 personal=np.concatenate([a[:,2,0,:,None],h.mean(1)],axis=-1)
 baseline=np.stack([a[:,2,0],a[:,0,0],a[:,1,0]],axis=-1)
 return personal,baseline,a[:,2,4]-a[:,2,0],h

def statistics(x,y):return np.stack([z.T@z for z in x]),np.stack([z.T@v for z,v in zip(x,y)])

def fit(x,y):
 g,r=statistics(x,y);return np.linalg.solve(g.sum(0)+np.eye(x.shape[-1]),r.sum(0))

def lopo(x,y):
 g,r=statistics(x,y);G=g.sum(0);R=r.sum(0);betas=[];pred=[]
 for i in range(len(x)):
  b=np.linalg.solve(G-g[i]+np.eye(x.shape[-1]),R-r[i]);betas.append(b);pred.append(x[i]@b)
 return np.array(pred),np.array(betas)

def cosine(u,v):
 den=float(np.linalg.norm(u)*np.linalg.norm(v))
 return (float(np.sum(u*v)/den),False) if den>0 else (0.,True)

def earlyscore(h):
 scores=[];flags=[]
 for i in range(len(h)):
  tr=np.arange(len(h))!=i;center=h[tr].mean(0);s,d=cosine(h[i,0]-center[0],h[i,1]-center[1]);scores.append(s);flags.append(d)
 return np.array(scores),flags

def losses(p,y):return np.mean((p-y)**2,axis=1)

def contrast(a,b):
 # candidate a versus reference b
 d=b-a;return {'candidate_mean_loss':float(a.mean()),'reference_mean_loss':float(b.mean()),'mean_reduction':float(d.mean()),'relative_mean_reduction':float(d.mean()/b.mean()),'participants_improved':int(np.sum(d>0)),'participant_differences':d.tolist()}
