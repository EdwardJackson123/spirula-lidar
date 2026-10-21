import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from spirula_lidar.sim3 import Similarity,umeyama,robust_similarity,transform_pose
from spirula_lidar.clouds import spatial_holdout,normals
from spirula_lidar.geometry import extract_planes,flatten,levels,refine,observability


def test_known_sim3_with_outliers(truth):
    rng=np.random.default_rng(9);x=rng.normal(size=(160,3));y=truth.apply(x)+rng.normal(scale=.001,size=x.shape);y[:40]+=8
    result,keep=robust_similarity(x,y,.02)
    assert keep.sum()==120
    assert abs(result.scale/truth.scale-1)<.001
    assert np.median(np.linalg.norm(result.apply(x)-truth.apply(x),axis=1))<.002


def test_uniform_similarity_rejects_shear_and_degeneracy():
    with pytest.raises(ValueError):Similarity(1,np.diag([1,2,1]),np.zeros(3))
    with pytest.raises(ValueError):Similarity(-1,np.eye(3),np.zeros(3))
    x=np.arange(10)[:,None]*np.ones((1,3))
    with pytest.raises(ValueError):umeyama(x,x)


def test_composition_inverse_and_pose_rotation(truth):
    normalize=Similarity(.37,Rotation.from_euler('x',31,degrees=True).as_matrix(),np.array([3,4,5]))
    x=np.array([[1,2,3.],[2,4,7.]])
    assert np.allclose(truth.compose(normalize.inverse()).apply(normalize.apply(x)),truth.apply(x))
    p=np.eye(4);p[:3,3]=x[0];q=transform_pose(p,truth)
    assert np.allclose(q[:3,:3].T@q[:3,:3],np.eye(3))
    assert np.allclose(q[:3,3],truth.apply(x[0]))


def test_spatial_holdout_keeps_same_cell_together():
    x=np.random.default_rng(2).normal(size=(6000,3));mask=spatial_holdout(x)
    assert .1<mask.mean()<.3
    assert np.array_equal(mask,spatial_holdout(x))
    shuffled=np.random.default_rng(1).permutation(len(x))
    assert np.array_equal(mask[shuffled],spatial_holdout(x[shuffled]))


def test_parallel_planes_arbitrary_rotation_and_origin():
    rng=np.random.default_rng(71);xy=rng.uniform(-3,3,(4000,2));x=np.c_[xy,np.r_[np.zeros(2000),np.full(2000,1.7)]]
    r=Rotation.from_euler('yx',[34,22],degrees=True).as_matrix();x=x@r.T+np.array([500.,-20.,77.])
    n=np.tile(r[:,2],(len(x),1));planes=extract_planes(x,n,.01);flat=flatten(planes[0]['normal']);ls=levels(planes,flat,.08)
    assert len(ls)==2
    assert abs(abs(ls[1]['height']-ls[0]['height'])-1.7)<.001


def test_coplanar_geometry_not_observable():
    rng=np.random.default_rng(2);x=np.c_[rng.uniform(-2,2,(2000,2)),np.zeros(2000)];n=np.tile([0.,0.,1.],(len(x),1))
    assert not observability(x,n,x,n,Similarity.identity())['full_rank']
