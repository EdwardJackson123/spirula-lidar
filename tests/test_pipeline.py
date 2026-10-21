import json
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from spirula_lidar import engine,pipeline
from spirula_lidar.alignment import reference_candidate,reference_validation,align_points
from spirula_lidar.artifacts import read,write,sha
from spirula_lidar.clouds import read_images,spatial_holdout
from spirula_lidar.sim3 import Similarity,transform_pose
from spirula_lidar.monitor import snapshot
from conftest import make_ply,make_sparse


def data_fixture(root,truth):
    rng=np.random.default_rng(72);points=rng.uniform([-2,-3,0],[2,3,2],(7000,3));poses={}
    for i in range(20):
        m=np.eye(4);m[:3,3]=[i*.11,np.sin(i*.2),1.];poses[f'{i:06d}.jpg']=m
    sparse=make_sparse(root/'sparse',points,poses)
    reference=root/'reference.json';write(reference,dict(frame='lidar',unit_to_m=1.,poses={n:transform_pose(p,truth) for n,p in poses.items()}))
    return points,poses,sparse,reference


def test_reference_pose_route_with_straight_path(tmp_path,truth):
    points,poses,sparse,reference=data_fixture(tmp_path,truth)
    # Orientations constrain rotation even when camera centers are collinear.
    for i,p in enumerate(poses.values()):p[:3,3]=[i*.2,0,1]
    sparse=make_sparse(sparse,points,poses);ref=read(reference);ref['poses']={n:transform_pose(p,truth) for n,p in poses.items()};write(reference,ref)
    result,details=reference_candidate(read_images(sparse/'images.bin'),reference,{})
    assert abs(result.scale/truth.scale-1)<1e-9
    checks,_=reference_validation(result,details);assert all(checks.values())


def test_reference_validation_does_not_fit_heldout(tmp_path,truth):
    points,poses,sparse,reference=data_fixture(tmp_path,truth);ref=read(reference)
    for name in sorted(poses)[::4]:ref['poses'][name][0][3]+=4
    write(reference,ref);result,details=reference_candidate(read_images(sparse/'images.bin'),reference,{})
    assert abs(result.scale/truth.scale-1)<1e-8
    checks,_=reference_validation(result,details);assert not all(checks.values())


def test_candidate_selection_sealed_before_geometry_validation(tmp_path,truth,monkeypatch):
    from spirula_lidar import alignment as module
    x=np.random.default_rng(38).normal(size=(2000,3));target=truth.apply(x);held=spatial_holdout(x)
    chosen=Similarity(1.7,np.eye(3),np.zeros(3));other=truth
    monkeypatch.setattr(module.geo,'surface_labels',lambda x,n:np.arange(len(x))%3)
    monkeypatch.setattr(module.geo,'parallel_plane_candidates',lambda *a:([
        dict(transform=chosen,group='a',proposal_score=1),dict(transform=other,group='b',proposal_score=1)],{}))
    monkeypatch.setattr(module.geo,'refine',lambda x,n,t,tn,tr,**kw:(tr,[]))
    monkeypatch.setattr(module.geo,'score',lambda x,n,l,t,tn,tr,**kw:dict(score=.9 if tr.scale==chosen.scale else .8,groups=[dict(coverage=.9)]*3))
    monkeypatch.setattr(module.geo,'observability',lambda *a:dict(full_rank=True))
    report=align_points(x,target,tmp_path/'alignment',dict(feature_fallback=False))
    assert report['similarity']['scale']==chosen.scale  # heldout would prefer truth; it cannot choose it.
    assert not report['passed']
    assert sha(tmp_path/'alignment/sealed_solution.json')==report['sealed_solution_sha256']


def test_rejected_alignment_never_exports(tmp_path):
    with pytest.raises(ValueError):pipeline.export_stage({},'unused',dict(passed=False),tmp_path,'unused','unused')
    assert not (tmp_path/'output').exists()


def test_complete_pipeline_with_simulated_engine_and_real_alignment(tmp_path,truth,monkeypatch):
    points,poses,sparse,reference=data_fixture(tmp_path,truth);images=tmp_path/'images';images.mkdir()
    for name in poses:Image.new('RGB',(16,12),'blue').save(images/name)
    lidar=tmp_path/'lidar.ply';make_ply(lidar,truth.apply(points))
    root=tmp_path/'run';normalize=Similarity(.4,Rotation.from_euler('z',29,degrees=True).as_matrix(),np.array([2.,3.,1.]))
    cfg=dict(images=str(images),lidar=str(lidar),sfm=str(sparse),reference_poses=str(reference),output=str(root),spirula=sys.executable,person_model=sys.executable,threads=2)
    monkeypatch.setattr(engine,'gpu_info',lambda i:dict(free_gib=30.,total_gib=32.,device='test'))
    counts=dict(masks=0,sfm=0,geometry=0,train=0)
    def masks(self):
        counts['masks']+=1;path=self.root/'test_masks.json';write(path,{});return {},[path]
    def sfm(self):
        counts['sfm']+=1;dest=make_sparse(self.data/'sparse/0',points,poses);return dict(points=len(points),registered_images=len(poses)),[dest/n for n in ('cameras.bin','images.bin','points3D.bin')]
    def geometry(self):
        counts['geometry']+=1;path=self.root/'test_geometry.json';write(path,{});return {},[path]
    def train(self,budget):
        counts['train']+=1;output=self.root/'training';output.mkdir();model=output/'native.ply';make_ply(model,normalize.apply(points[:6]),True)
        path=output/'scene_transform.json';write(path,{'train_from_world':normalize.as_dict()});return dict(model=str(model),scene_transform=str(path)),[model,path]
    for key,fn in dict(masks=masks,sfm=sfm,geometry=geometry,train=train).items():monkeypatch.setattr(pipeline.Job,key,fn)
    assert pipeline.run(cfg)==0
    assert read(root/'status.json')['status']=='completed'
    v=PlyData.read(root/'output/scene.ply')['vertex'].data;actual=np.column_stack([v[n] for n in ('x','y','z')])
    assert np.allclose(actual,truth.apply(points[:6]),atol=1e-5)
    assert pipeline.run(cfg)==0
    assert counts==dict(masks=1,sfm=1,geometry=1,train=1)
    Image.new('RGB',(16,12),'red').save(images/next(iter(poses)))
    with pytest.raises(ValueError,match='changed'):pipeline.run(cfg)


def test_training_flags_preserve_image_only_defaults(tmp_path):
    args=engine.training_command('spirula',tmp_path/'dataset',tmp_path,dict(iterations=80000,gaussians=4000000,cache_images='cpu'),'auto')
    assert args[args.index('--sh-degree')+1]=='0'
    assert args[args.index('--depth-supervision-weight')+1]=='.01'
    assert args[args.index('--normal-supervision-weight')+1]=='.01'
    assert args[args.index('--train-resolution-divisor')+1]=='1'
    assert args[args.index('--floater-suppression')+1]=='mild'
    assert not any('lidar' in a.lower() for a in args)


def test_monitor_cannot_read_log_outside_job(tmp_path):
    root=tmp_path/'run';root.mkdir();secret=tmp_path/'private';secret.write_text('do not serve')
    write(root/'status.json',dict(status='running',log=str(secret),updated_at=123))
    assert snapshot(root)==dict(status=dict(status='running',log=str(secret),updated_at=123),log='')


def test_insufficient_memory_rejected():
    with pytest.raises(MemoryError):engine.plan_budget(dict(points=500000,registered_images=500),[dict(width=1920,height=1080)],dict(free_gib=3,total_gib=32),{})


def test_dry_run_can_be_followed_by_real_run(tmp_path,monkeypatch):
    images=tmp_path/'images';images.mkdir()
    for i in range(12):Image.new('RGB',(16,12)).save(images/f'{i}.jpg')
    lidar=tmp_path/'lidar.ply';make_ply(lidar,np.random.default_rng(1).normal(size=(600,3)))
    cfg=dict(images=str(images),lidar=str(lidar),output=str(tmp_path/'run'),spirula=sys.executable,dry_run=True)
    assert pipeline.run(cfg)==0
    saved=read(tmp_path/'run/job.json')['fingerprint'];cfg.pop('dry_run')
    monkeypatch.setattr(pipeline,'Job',lambda *a:(_ for _ in ()).throw(RuntimeError('reached worker')))
    with pytest.raises(RuntimeError,match='reached worker'):pipeline.run(cfg)
    assert read(tmp_path/'run/job.json')['fingerprint']==saved
