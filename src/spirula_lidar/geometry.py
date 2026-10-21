"""Fit-only candidate generation/refinement; no validation points enter here."""
import math
import numpy as np
from scipy import fft
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .sim3 import Similarity
from .clouds import normals


def extent(x): return float(np.linalg.norm(np.quantile(x,.99,axis=0)-np.quantile(x,.01,axis=0)))


def extract_planes(x,n,tolerance,seed=19):
    axis=np.linalg.eigh(n.T@n)[1][:,-1];axis*=np.sign(axis[np.argmax(np.abs(axis))])
    remaining=x[np.abs(n@axis)>.94];rng=np.random.default_rng(seed);planes=[]
    for _ in range(9):
        if len(remaining)<100: break
        sample=remaining[rng.choice(len(remaining),min(4000,len(remaining)),replace=False)]
        triples=sample[rng.integers(len(sample),size=(2500,3))]
        nn=np.cross(triples[:,1]-triples[:,0],triples[:,2]-triples[:,0]);length=np.linalg.norm(nn,axis=1)
        ok=length>1e-8;nn=nn[ok]/length[ok,None];pp=triples[ok,0]
        ok=np.abs(nn@axis)>.97;nn,pp=nn[ok],pp[ok]
        if not len(nn): break
        offsets=-np.sum(nn*pp,axis=1); best=None
        for start in range(0,len(nn),128):
            distance=np.abs(sample@nn[start:start+128].T+offsets[start:start+128])
            support=(distance<tolerance).sum(0);i=int(np.argmax(support))+start
            if best is None or support.max()>best[0]: best=(int(support.max()),i)
        i=best[1];good=np.abs(remaining@nn[i]+offsets[i])<tolerance
        if good.sum()<max(100,.002*len(x)): break
        points=remaining[good];center=points.mean(0);normal=np.linalg.svd(points-center,full_matrices=False)[2][-1]
        normal*=np.sign(normal@axis);offset=-normal@center
        good=np.abs(remaining@normal+offset)<tolerance; points=remaining[good]
        planes.append(dict(normal=normal,offset=float(offset),support=len(points),centroid=points.mean(0)))
        remaining=remaining[~good]
    return planes


def flatten(axis):
    axis=np.asarray(axis,float).copy();axis/=np.linalg.norm(axis)
    return Rotation.align_vectors(np.array([[0.,0.,1.]]),axis[None,:])[0].as_matrix()


def levels(planes,rotation,merge):
    result=[]
    for plane in sorted(planes,key=lambda p:p['support'],reverse=True):
        h=float(plane['centroid']@rotation[2])
        if not any(abs(h-p['height'])<merge for p in result): result.append(dict(height=h,support=plane['support']))
    return result[:6]


def parallel_plane_candidates(x,n,target,tn,cfg,progress):
    source_tolerance=extent(x)/1000;target_tolerance=max(.025,cfg.get('lidar_voxel_m',.08)*.3)
    a=extract_planes(x,n,source_tolerance);b=extract_planes(target,tn,target_tolerance)
    if len(a)<2 or len(b)<2: return [],dict(reason='Insufficient parallel-plane structure')
    sr=flatten(a[0]['normal']);tr=flatten(sum(p['support']*p['normal'] for p in b[:2]))
    sl=levels(a,sr,source_tolerance*10);tl=levels(b,tr,target_tolerance*4)
    if len(sl)<2 or len(tl)<2: return [],dict(reason='Insufficient separated horizontal levels')
    center=(x@sr.T).mean(0);sx=x@sr.T-center;tx=target@tr.T
    hypotheses=[]
    for i,sa in enumerate(sl):
        for sb in sl[:i]:
            slo,shi=sorted((sa['height'],sb['height']));ds=shi-slo
            if ds<source_tolerance*20:continue
            for j,ta in enumerate(tl):
                for tb in tl[:j]:
                    tlo,thi=sorted((ta['height'],tb['height']));dt=thi-tlo
                    if dt<max(.3,target_tolerance*8):continue
                    scale=dt/ds
                    if not .5<extent(x)*scale<extent(target)*3:continue
                    weight=math.sqrt(sa['support']*sb['support']*ta['support']*tb['support'])
                    hypotheses.append(dict(scale=scale,slo=slo-center[2],shi=shi-center[2],tlo=tlo,thi=thi,weight=weight))
    chosen=[]
    for h in sorted(hypotheses,key=lambda h:h['weight'],reverse=True):
        if any(abs(math.log(h['scale']/q['scale']))<.025 and abs(h['tlo']-q['tlo'])<.1 for q in chosen):continue
        chosen.append(h)
        if len(chosen)>=cfg.get('max_scale_hypotheses',8):break
    if not chosen:return [],dict(reason='No geometric scale hypotheses')
    size=512
    step=max(cfg.get('fft_voxel_m',.16),np.ptp(tx[:,:2],axis=0).max()/220,
             max(h['scale'] for h in chosen)*np.abs(sx[:,:2]).max()/190)
    origin=tx[:,:2].min(0)-128*step
    # Bins are relative to this map's observed heights, not its arbitrary Z origin.
    zlo,zhi=np.quantile(tx[:,2],[.005,.995]);edges=np.r_[-np.inf,np.linspace(zlo,zhi,7)[1:-1],np.inf]
    channels=len(edges)-1;vertical=np.abs((tn@tr.T)[:,2])<.75;vx=tx[vertical]
    indices=np.floor((vx[:,:2]-origin)/step).astype(int);fields=[]
    for k in range(channels):
        select=(vx[:,2]>=edges[k])&(vx[:,2]<edges[k+1]);mask=np.zeros((size,size),bool)
        mask[indices[select,0],indices[select,1]]=True
        field=np.exp(-.5*(distance_transform_edt(~mask)*step/max(.13,step*.8))**2) if mask.any() else np.zeros_like(mask,dtype=float)
        fields.append(field.astype(np.float32))
    threads=cfg.get('threads',4);tf=fft.rfft2(np.array(fields),workers=threads)
    source=sx[np.abs((n@sr.T)[:,2])<.75]
    shifts=np.arange(size);signed=np.where(shifts<size//2,shifts,shifts-size)
    coords=origin[:,None]+(signed[None,:]+size//2)*step
    lo,hi=tx[:,:2].min(0)-step*2,tx[:,:2].max(0)+step*2
    allowed=(coords[0,:,None]>=lo[0])&(coords[0,:,None]<=hi[0])&(coords[1,None,:]>=lo[1])&(coords[1,None,:]<=hi[1])
    all_candidates=[]
    for hidx,h in enumerate(chosen):
        for sign in (1,-1):
            progress('global_alignment',hypothesis=hidx+1,total_hypotheses=len(chosen),vertical_sign=sign,scale=h['scale'])
            flip=np.diag([1.,float(sign),float(sign)]);zshift=h['tlo']-h['scale']*h['slo'] if sign==1 else h['thi']+h['scale']*h['slo']
            rows=[]
            for angle in range(0,360,cfg.get('yaw_step_degrees',4)):
                r=Rotation.from_euler('z',angle,degrees=True).as_matrix()@flip
                q=h['scale']*source@r.T;q[:,2]+=zshift;ij=np.rint(q[:,:2]/step).astype(int)+size//2
                if (ij<0).any() or (ij>=size).any():continue
                image=np.zeros((channels,size,size),np.float32);occupied=0
                for k in range(channels):
                    select=(q[:,2]>=edges[k])&(q[:,2]<edges[k+1]);count=int(select.sum())
                    if count<40:continue
                    np.add.at(image[k],(ij[select,0],ij[select,1]),1/count);occupied+=1
                if occupied<2:continue
                correlation=fft.irfft2(np.sum(tf*np.conj(fft.rfft2(image,workers=threads)),axis=0),s=(size,size),workers=threads)/occupied
                correlation[~allowed]=-1
                for _ in range(2):
                    i,j=np.unravel_index(int(correlation.argmax()),correlation.shape);xy=origin+(np.array([signed[i],signed[j]])+size//2)*step
                    local_t=np.r_[xy,zshift];native_r=tr.T@r@sr;native_t=tr.T@(local_t-h['scale']*r@center)
                    rows.append(dict(transform=Similarity(h['scale'],native_r,native_t),group=f'plane-{hidx}-{sign}',proposal_score=float(correlation[i,j])))
                    correlation[np.ix_(np.arange(i-5,i+6)%size,np.arange(j-5,j+6)%size)]=-1
            pilot=x[::max(1,len(x)//500)];selected=[]
            for row in sorted(rows,key=lambda r:r['proposal_score'],reverse=True):
                query=row['transform'].apply(pilot)
                if any(np.sqrt(np.mean(np.sum((query-p['transform'].apply(pilot))**2,axis=1)))<max(.45,step*2) for p in selected):continue
                selected.append(row)
                if len(selected)>=cfg.get('coarse_per_mode',24):break
            all_candidates.extend(selected)
    return all_candidates,dict(source_planes=a,target_planes=b,source_levels=sl,target_levels=tl,scale_hypotheses=chosen,xy_voxel_m=step,height_edges=edges[1:-1])


def surface_labels(x,n):
    axis=np.linalg.eigh(n.T@n)[1][:,-1];h=np.abs(n@axis)>.9;z=x@axis
    # The two most supported separated plane levels determine the split.
    ps=extract_planes(x,n,extent(x)/1000);r=flatten(ps[0]['normal']) if ps else flatten(axis)
    ls=levels(ps,r,extent(x)/100) if ps else []
    split=(ls[0]['height']+ls[1]['height'])/2 if len(ls)>1 else float(np.median(z[h]))
    if len(ls)>1:z=x@r[2]
    labels=np.ones(len(x),int);labels[h&(z<split)]=0;labels[h&(z>=split)]=2
    return labels


def score(x,n,labels,tree,tn,transform,tolerance=.12,threads=4):
    q=transform.apply(x);distance,ids=tree.query(q,workers=threads)
    normal=np.abs(np.sum((n@transform.rotation.T)*tn[ids],axis=1))>.70;rows=[]
    for group in np.unique(labels):
        selected=labels==group
        if selected.sum()<100:continue
        rows.append(dict(group=int(group),points=int(selected.sum()),coverage=float(np.mean((distance[selected]<tolerance)&normal[selected])),median_m=float(np.median(distance[selected]))))
    if not rows:raise ValueError('No supported surface group')
    return dict(score=float(np.mean([r['coverage'] for r in rows])-.15*np.mean(np.minimum(distance,.4))/.4),groups=rows,median_m=float(np.median(distance)),within10cm=float(np.mean(distance<.1)))


def refine(x,n,target,tn,initial,threads=4,coarse_gate=.5):
    tree=cKDTree(target);s=initial.scale;r=initial.rotation.copy();t=initial.translation.copy();history=[]
    for gate,steps in ((coarse_gate,25),(.25,30),(.14,35)):
        for iteration in range(steps):
            q=s*x@r.T+t;d,ids=tree.query(q,workers=threads);nn=tn[ids]
            keep=(d<gate)&(np.abs(np.sum((n@r.T)*nn,axis=1))>.65)
            if keep.sum()<100:raise ValueError('Insufficient geometric support')
            v=q[keep];nn=nn[keep];res=np.sum((v-target[ids[keep]])*nn,axis=1)
            trim=np.abs(res)<=min(gate*.65,max(.04,float(np.quantile(np.abs(res),.85))))
            v,nn,res=v[trim],nn[trim],res[trim];pivot=v.mean(0);centered=v-pivot
            j=np.column_stack((np.sum(centered*nn,axis=1),np.cross(centered,nn),nn));weights=1/np.sqrt(1+(res/.03)**2)
            delta=np.linalg.lstsq(j*weights[:,None],-res*weights,rcond=1e-7)[0]
            delta[0]=np.clip(delta[0],-.025,.025)
            for section,cap in ((slice(1,4),.05),(slice(4,7),.10)):
                length=np.linalg.norm(delta[section])
                if length>cap:delta[section]*=cap/length
            factor=np.exp(delta[0]);dr=Rotation.from_rotvec(delta[1:4]).as_matrix()
            t=factor*dr@(t-pivot)+pivot+delta[4:];r=dr@r;s*=factor
            history.append(dict(gate=gate,iteration=iteration,scale=s,pairs=len(res),plane_rmse_m=float(np.sqrt(np.mean(res**2)))))
            if np.linalg.norm(delta)<3e-6:break
    return Similarity(s,r,t),history


def observability(x,n,target,tn,transform,threads=4):
    q=transform.apply(x);d,ids=cKDTree(target).query(q,workers=threads)
    keep=(d<.14)&(np.abs(np.sum((n@transform.rotation.T)*tn[ids],axis=1))>.7)
    if keep.sum()<100:return dict(full_rank=False,pairs=int(keep.sum()))
    v=q[keep];v-=v.mean(0);nn=tn[ids[keep]]
    j=np.column_stack((np.sum(v*nn,axis=1),np.cross(v,nn),nn));length=np.linalg.norm(j,axis=0)
    singular=np.linalg.svd(j/np.maximum(length,1e-12),compute_uv=False)
    return dict(full_rank=bool(singular[-1]>.02 and length.min()>1e-6),pairs=int(keep.sum()),normalized_singular_values=singular,condition=float(singular[0]/max(singular[-1],1e-15)))
