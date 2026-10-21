"""Route selection, frozen fit-only solution, and separate acceptance checks."""
import math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from . import geometry as geo
from .artifacts import read,write,sha,status
from .clouds import read_images,spatial_holdout,normals
from .sim3 import Similarity,c2w_from_record,robust_similarity


def ranking_key(row):
    # Prefer support on the weakest surface group, then the overall score.
    return min(r['coverage'] for r in row['fit']['groups']),row['fit']['score']


def load_reference(path):
    path=Path(path)
    if path.is_dir():
        return {name:c2w_from_record(row) for name,row in read_images(path/'images.bin').items()},1.
    data=read(path)
    if data.get('frame')!='lidar': raise ValueError('Reference poses must declare frame="lidar"')
    unit=float(data.get('unit_to_m',1.))
    if not np.isfinite(unit) or unit<=0: raise ValueError('Invalid reference pose units')
    poses={}
    for name,value in data['poses'].items():
        m=np.asarray(value,float)
        if m.shape!=(4,4) or not np.isfinite(m).all() or not np.allclose(m[3],[0,0,0,1]):raise ValueError('Invalid reference c2w: '+name)
        Similarity(1,m[:3,:3],m[:3,3]);poses[name]=m
    return poses,unit


def reference_candidate(source_images,path,cfg):
    reference,unit=load_reference(path);names=sorted(set(source_images)&set(reference))
    if len(names)<12:raise ValueError('Reference route needs at least 12 matching image names')
    source=np.array([c2w_from_record(source_images[n]) for n in names]);target=np.array([reference[n] for n in names]);target[:,:3,3]*=unit
    held=np.arange(len(names))%4==0;fit=~held;ids=np.flatnonzero(fit);rng=np.random.default_rng(911)
    rotations=target[fit,:3,:3]@source[fit,:3,:3].transpose(0,2,1)
    angles=np.stack([Rotation.from_matrix(rotations@r.T).magnitude() for r in rotations])
    consensus=angles[int(np.argmax((angles<np.deg2rad(3)).sum(1)))]<np.deg2rad(3)
    if consensus.sum()<6:raise ValueError('Reference orientations lack a common rotation')
    r=Rotation.from_matrix(rotations[consensus]).mean().as_matrix();x=source[:,:3,3];y=target[:,:3,3]
    if np.linalg.norm(np.ptp(x[fit],axis=0))<1e-5:raise ValueError('One camera position does not determine scale')
    best=None
    for _ in range(1000):
        a,b=rng.choice(ids,2,replace=False);dx=r@(x[b]-x[a]);dy=y[b]-y[a];den=dx@dx
        if den<1e-8:continue
        s=float(dx@dy/den)
        if s<=0:continue
        t=y[[a,b]].mean(0)-s*r@x[[a,b]].mean(0);d=np.linalg.norm(s*x@r.T+t-y,axis=1);keep=fit&(d<.2)
        if best is None or keep.sum()>best.sum():best=keep
    if best is None or best.sum()<6:raise ValueError('Reference centers lack consensus')
    for _ in range(5):
        xx=x[best]-x[best].mean(0);yy=y[best]-y[best].mean(0);s=float(np.sum((xx@r.T)*yy)/np.sum(xx*xx))
        t=y[best].mean(0)-s*r@x[best].mean(0);keep=fit&(np.linalg.norm(s*x@r.T+t-y,axis=1)<.2)
        if keep.sum()<6 or np.array_equal(keep,best):break
        best=keep
    transform=Similarity(s,r,t)
    # Only fit anchors contribute to scale stability estimation.
    scales=[]
    for _ in range(100):
        pick=rng.choice(np.flatnonzero(best),size=max(4,int(best.sum()*.8)),replace=False)
        a=x[pick]-x[pick].mean(0);b=y[pick]-y[pick].mean(0);scales.append(float(np.sum((a@r.T)*b)/np.sum(a*a)))
    spread=float((np.quantile(scales,.95)-np.quantile(scales,.05))/s*100)
    return transform,dict(names=names,source=source,target=target,heldout=held,scale_spread_percent=spread,
                         source_reference=str(path),reference_unit_to_m=unit,uses_reference_poses=True)


def reference_validation(transform,details):
    source,target=details['source'],details['target'];held=details['heldout']
    distance=np.linalg.norm(transform.apply(source[:,:3,3])-target[:,:3,3],axis=1)
    angle=Rotation.from_matrix(target[:,:3,:3].transpose(0,2,1)@transform.rotation@source[:,:3,:3]).magnitude()*180/np.pi
    h=distance[held];a=angle[held]
    checks=dict(reference_heldout_views=len(h)>=3,reference_position_median_le_15cm=float(np.median(h))<=.15,
                reference_orientation_median_le_3deg=float(np.median(a))<=3,
                reference_within_20cm_5deg_ge_80percent=float(np.mean((h<.2)&(a<5)))>=.8,
                fit_anchor_scale_spread_le_2percent=details['scale_spread_percent']<=2)
    metrics=dict(heldout_views=len(h),position_median_m=float(np.median(h)),position_p90_m=float(np.percentile(h,90)),orientation_median_deg=float(np.median(a)),scale_spread_percent=details['scale_spread_percent'])
    return checks,metrics


def feature_candidates(x,target,cfg,progress):
    try: import open3d as o
    except ImportError:return [],dict(reason='No parallel planes; install [features] or provide reference poses/landmarks')
    reg=o.pipelines.registration;ca=np.median(x,axis=0);cb=np.median(target,axis=0);ratio=geo.extent(target)/geo.extent(x)
    voxel=max(.16,geo.extent(target)/200);rows=[]
    def features(p):
        c=o.geometry.PointCloud();c.points=o.utility.Vector3dVector(p);c=c.voxel_down_sample(voxel)
        c.estimate_normals(o.geometry.KDTreeSearchParamHybrid(radius=voxel*2,max_nn=30))
        return c,reg.compute_fpfh_feature(c,o.geometry.KDTreeSearchParamHybrid(radius=voxel*5,max_nn=100))
    tp,tf=features(target-cb)
    for factor in (.25,.5,.75,1.,1.5,2.):
        s=ratio*factor;sp,sf=features((x-ca)*s)
        for seed in (19,47):
            progress('feature_global_registration',scale=s,seed=seed);o.utility.random.seed(seed)
            res=reg.registration_ransac_based_on_feature_matching(sp,tp,sf,tf,True,voxel*1.5,
                reg.TransformationEstimationPointToPoint(False),3,[reg.CorrespondenceCheckerBasedOnEdgeLength(.85),reg.CorrespondenceCheckerBasedOnDistance(voxel*1.5)],reg.RANSACConvergenceCriteria(150000,.999))
            if res.fitness<.1:continue
            r=np.asarray(res.transformation)[:3,:3];t=cb+np.asarray(res.transformation)[:3,3]-s*r@ca
            rows.append(dict(transform=Similarity(s,r,t),group=f'feature-{factor}-{seed}',proposal_score=float(res.fitness)))
    return rows,dict(method='Intrinsic-extent hypotheses + FPFH global RANSAC; finite search')


def align_points(source,target,output,cfg=None,source_images=None,reference_poses=None,landmarks=None,progress=None):
    cfg=cfg or {};output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'sealed_solution.json').exists():raise FileExistsError('Use a fresh alignment directory; an immutable solution is already present')
    progress=progress or (lambda stage,**kw:status(output,stage,**kw))
    source,target=np.asarray(source,float),np.asarray(target,float)
    if source.ndim!=2 or source.shape[1]!=3 or target.ndim!=2 or target.shape[1]!=3 or not np.isfinite(source).all() or not np.isfinite(target).all():raise ValueError('Point clouds must be finite Nx3')
    held=spatial_holdout(source);fit=source[~held];threads=cfg.get('threads',4)
    progress('estimate_normals',fit_points=len(fit),lidar_points=len(target))
    sn=normals(fit,threads=threads);tn=normals(target,threads=threads);labels=geo.surface_labels(fit,sn);tree=cKDTree(target)
    method='direct_geometry';details={};selected=None;coarse=[];refined=[];attempts=[]
    if reference_poses:
        if source_images is None:raise ValueError('Reference route needs SfM camera poses')
        try:
            transform,details=reference_candidate(source_images,reference_poses,cfg)
            selected=dict(transform=transform,fit=geo.score(fit,sn,labels,tree,tn,transform,threads=threads));method='reference_poses'
        except ValueError as error:attempts.append(dict(route='reference_poses',reason=str(error)))
    if selected is None:
        if landmarks:
            pairs=read(landmarks);a=np.asarray(pairs['sfm_xyz']);b=np.asarray(pairs['lidar_xyz'])*float(pairs.get('lidar_unit_to_m',1.))
            transform,keep=robust_similarity(a,b,.15)
            coarse=[dict(transform=transform,group='landmarks',proposal_score=int(keep.sum()))];details=dict(method='User-provided 3D correspondences',uses_landmarks=True)
        else:
            coarse,details=geo.parallel_plane_candidates(fit,sn,target,tn,cfg,progress)
            if not coarse and cfg.get('feature_fallback',True):
                coarse,feature_details=feature_candidates(fit,target,cfg,progress);details.update(feature_details)
        write(output/'proposals.json',dict(details=details,candidates=[{**r,'transform':r['transform'].as_dict()} for r in coarse]))
        groups={}
        for row in coarse:
            row['fit']=geo.score(fit,sn,labels,tree,tn,row['transform'],threads=threads);groups.setdefault(row['group'],[]).append(row)
        shortlist=[]
        for rows in groups.values():shortlist.extend(sorted(rows,key=ranking_key,reverse=True)[:cfg.get('refine_per_mode',4)])
        for i,row in enumerate(shortlist):
            progress('refine_alignment',candidate=i+1,total=len(shortlist))
            try:
                transform,history=geo.refine(fit,sn,target,tn,row['transform'],threads=threads,coarse_gate=max(.5,details.get('xy_voxel_m',.16)*2))
                valid=abs(math.log(transform.scale/row['transform'].scale))<=.10
                refined.append(dict(transform=transform,group=row['group'],initial_scale=row['transform'].scale,valid=valid,
                                    fit=geo.score(fit,sn,labels,tree,tn,transform,threads=threads),history=history))
            except ValueError as error:attempts.append(dict(route=row['group'],reason=str(error)))
        valid=[r for r in refined if r['valid']]
        if valid:selected=max(valid,key=ranking_key)
    if selected is None:
        report=dict(passed=False,status='needs_review',method=method,reason='No valid global hypothesis. Supply matching reference poses or distributed 3D correspondences.',attempts=attempts,uses_reference_poses=False,global_uniqueness_proven=False)
        write(output/'alignment.json',report);return report
    transform=selected['transform'];observable=geo.observability(fit,sn,target,tn,transform,threads)
    pilot=fit[::max(1,len(fit)//500)];base=transform.apply(pilot);alternatives=[]
    for row in refined:
        if not row['valid']:continue
        rms=float(np.sqrt(np.mean(np.sum((row['transform'].apply(pilot)-base)**2,axis=1))))
        if rms>.25:alternatives.append(dict(score=row['fit']['score'],weakest_surface_coverage=ranking_key(row)[0],scale=row['transform'].scale,rms_displacement_m=rms))
    gap=ranking_key(selected)[0]-max(r['weakest_surface_coverage'] for r in alternatives) if alternatives else None
    sealed=dict(similarity=transform.as_dict(),method=method,fit=selected['fit'],observability=observable,
                reference_fit=details if method=='reference_poses' else None,
                refined=[{**r,'transform':r['transform'].as_dict()} for r in refined],tested_alternatives=alternatives,
                tested_alternative_gap=gap,alternative_gap_metric='weakest surface-group coverage',selection='fit data only; weakest group first, then mean score; validation distances not consulted',
                heldout_indices=np.flatnonzero(held),attempts=attempts,global_uniqueness_proven=False)
    write(output/'sealed_solution.json',sealed);sealed_sha=sha(output/'sealed_solution.json')
    # First access to validation geometry after immutable candidate selection.
    distance=tree.query(transform.apply(source[held]),workers=threads)[0]
    metrics=dict(points=len(distance),median_m=float(np.median(distance)),p90_m=float(np.percentile(distance,90)),within10cm=float(np.mean(distance<.1)))
    checks=dict(heldout_median_le_5cm=metrics['median_m']<=cfg.get('max_heldout_median_m',.05),
                heldout_within10cm_ge_70percent=metrics['within10cm']>=cfg.get('min_heldout_10cm_fraction',.70))
    reference_metrics=None
    if method=='reference_poses':
        rc,reference_metrics=reference_validation(transform,details);checks.update(rc)
    else:
        checks.update(surface_group_coverage=bool(len(selected['fit']['groups'])>=3 and all(r['coverage']>=cfg.get('min_surface_group_coverage',.65) for r in selected['fit']['groups'])),
                      sim3_observable=observable['full_rank'],tested_alternative_gap=bool(gap is not None and gap>=.03))
    passed=all(checks.values())
    report=dict(passed=passed,status='accepted' if passed else 'needs_review',method=method,similarity=transform.as_dict(),fit=selected['fit'],heldout=metrics,
                reference_metrics=reference_metrics,checks=checks,observability=observable,sealed_solution_sha256=sealed_sha,
                uses_reference_poses=method=='reference_poses',uses_lidar_in_training=False,uses_lidar_for_alignment=True,
                geometric_surface_check='passed' if all(r['coverage']>=.65 for r in selected['fit']['groups']) else 'incomplete',
                absolute_survey_accuracy_verified=False,scale_error_percent=None,global_uniqueness_proven=False,
                certification='not_certified',units='metres relative to declared LiDAR units',
                reason='Passed route-specific checks' if passed else 'Strict alignment checks failed; metric model is not exported')
    write(output/'alignment.json',report)
    _preview(output,source,target,held,transform,distance)
    return read(output/'alignment.json')


def _preview(output,source,target,held,transform,distance):
    # Portable self-contained HTML with XY/XZ/YZ projections, no CDN or server.
    import json
    q=transform.apply(source);q=q[::max(1,len(q)//12000)];t=target[::max(1,len(target)//18000)]
    payload=json.dumps(dict(source=q.tolist(),target=t.tolist()),separators=(',',':'))
    html='''<!doctype html><meta charset="utf-8"><title>Alignment geometry</title>
<style>body{font:15px system-ui;margin:24px;background:#fafafa}canvas{background:white;border:1px solid #ccc;max-width:100%;margin:6px}p{max-width:900px}</style>
<h1>Candidate geometry alignment</h1><p>Grey: measured LiDAR. Orange: transformed image SfM. Consult alignment.json for acceptance; this overlay does not certify scale or local geometry.</p><div id="views"></div><script>
const data=PAYLOAD;for(const [a,b,label] of [[0,1,'XY'],[0,2,'XZ'],[1,2,'YZ']]){let c=document.createElement('canvas');c.width=600;c.height=500;document.querySelector('#views').append(c);let ctx=c.getContext('2d'),all=data.source.concat(data.target);let lo=[Math.min(...all.map(p=>p[a])),Math.min(...all.map(p=>p[b]))],hi=[Math.max(...all.map(p=>p[a])),Math.max(...all.map(p=>p[b]))];let s=Math.min(550/(hi[0]-lo[0]),440/(hi[1]-lo[1]));ctx.fillText(label+' (metres)',20,20);for(const [name,color] of [['target','#8b959e66'],['source','#e77c20aa']]){ctx.fillStyle=color;for(const p of data[name])ctx.fillRect(25+(p[a]-lo[0])*s,470-(p[b]-lo[1])*s,1.3,1.3);}};</script>'''.replace('PAYLOAD',payload)
    (output/'preview.html').write_text(html,encoding='utf8')
