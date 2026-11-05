import numpy as np, sys, csv, time
from scipy.integrate import solve_ivp
from multiprocessing import Pool
BETA,GAMMA,OMEGA,I0=0.2,0.1,0.025,1e-6
T,TAU,TEND=250.0,50.0,5000.0
ATOL=[1e-10,1e-10,1e-60,1e-10]
def rhs(t,y,mode,a):
    c,s,i,r=y; G=a if mode=='alpha' else a*i
    return [G*s-OMEGA*c, OMEGA*c-G*s-BETA*s*i, BETA*s*i-GAMMA*i, GAMMA*i]
def R(mode,a,t1,f,gap):
    p1=(t1,t1+f*TAU); p2=(p1[1]+gap,p1[1]+gap+(1-f)*TAU) if f<1 else None
    segs=[]; cur=0.0
    for (s,e) in [p for p in (p1,p2) if p]:
        s2=min(s,T); e2=min(e,T)
        if s2>=T: break
        if cur<s2: segs.append((cur,s2,0))
        if e2>s2: segs.append((s2,e2,1)); cur=e2
        else: cur=max(cur,s2)
    if cur<TEND: segs.append((cur,TEND,0))
    y=np.array([0,1-I0,I0,0.0])
    for s,e,on in segs:
        if e<=s: continue
        y=solve_ivp(rhs,(s,e),y,method='LSODA',rtol=1e-8,atol=ATOL,args=(mode,a if on else 0.0)).y[:,-1]
    return y[3]
def opt(args):
    mode,a,f,gap=args
    ts=np.arange(0,T+1e-9,2.0)
    v=[R(mode,a,t,f,gap) for t in ts]; k=int(np.argmin(v))
    tf=np.arange(max(0,ts[k]-2),min(T,ts[k]+2)+1e-9,0.1)
    vf=[R(mode,a,t,f,gap) for t in tf]; j=int(np.argmin(vf))
    return (mode,a,f,gap,float(tf[j]),float(vf[j]))
if __name__=='__main__':
    cases=[('alpha',0.005),('alpha',0.01),('alpha',0.5),('eta',0.25),('eta',0.35),('eta',1.0)]
    jobs=[(m,a,1.0,0.0) for m,a in cases]
    jobs+=[(m,a,f,g) for m,a in cases for f in np.round(np.arange(0.1,0.95,0.1),1) for g in np.arange(5.0,200.1,5.0)]
    t0=time.time(); out=[]
    with Pool(2) as p, open('verify_tau50.csv','w') as fh:
        w=csv.writer(fh)
        for k,r in enumerate(p.imap_unordered(opt,jobs,chunksize=4)):
            w.writerow(r); fh.flush()
            if k%50==0: print(k,len(jobs),round(time.time()-t0),flush=True)
    print('done',round(time.time()-t0),flush=True)
