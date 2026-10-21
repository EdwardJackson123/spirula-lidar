import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from spirula_lidar.geometry import parallel_plane_candidates,refine
from spirula_lidar.clouds import normals
from spirula_lidar.sim3 import Similarity
from spirula_lidar.alignment import align_points


def test_global_plane_search_recovers_scale_without_initial_transform(tmp_path):
    rng=np.random.default_rng(118);pieces=[]
    for z in (0.,2.):pieces.append(np.c_[rng.uniform(-2,2,2000),rng.uniform(-3,3,2000),np.full(2000,z)])
    for x in (-2.,2.):pieces.append(np.c_[np.full(700,x),rng.uniform(-3,3,700),rng.uniform(0,2,700)])
    for y in (-3.,3.):pieces.append(np.c_[rng.uniform(-2,2,700),np.full(700,y),rng.uniform(0,2,700)])
    for cx,cy in [(-1.,.8),(.6,1.7)]:
        angle=rng.uniform(0,2*np.pi,600);pieces.append(np.c_[cx+.2*np.cos(angle),cy+.2*np.sin(angle),rng.uniform(.1,1.7,600)])
    source=np.concatenate(pieces)
    truth=Similarity(2.37,Rotation.from_euler('zyx',[41,7,-4],degrees=True).as_matrix(),np.array([400.,-500.,77.]))
    target=truth.apply(source);sn=normals(source,threads=2);tn=normals(target,threads=2)
    candidates,meta=parallel_plane_candidates(source,sn,target,tn,dict(threads=2,max_scale_hypotheses=2,yaw_step_degrees=8,coarse_per_mode=6),lambda *a,**k:None)
    assert candidates
    tree=cKDTree(target)
    best=min(candidates,key=lambda row:np.mean(np.minimum(tree.query(row['transform'].apply(source),workers=2)[0],.5)))
    fitted,_=refine(source,sn,target,tn,best['transform'],threads=2)
    assert abs(fitted.scale/truth.scale-1)<.01
    assert np.median(tree.query(fitted.apply(source),workers=2)[0])<.025
    report=align_points(source,target,tmp_path/'geometry_only',dict(threads=2,max_scale_hypotheses=2,yaw_step_degrees=8,coarse_per_mode=6,refine_per_mode=2,feature_fallback=False))
    assert report['passed'],report['checks']
    assert not report['uses_reference_poses']
    assert abs(report['similarity']['scale']/truth.scale-1)<.01
