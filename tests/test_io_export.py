import json
import numpy as np
import pytest
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from spirula_lidar.gaussians import transform_ply,model_to_map,transform_colmap
from spirula_lidar.sim3 import Similarity,c2w_from_record
from spirula_lidar.clouds import read_images,read_sfm_points,read_lidar,lidar_chunks
from spirula_lidar.artifacts import write,disjoint,stage
from conftest import make_ply,make_sparse


def test_gaussian_centers_covariances_and_colors(tmp_path,truth):
    xyz=np.random.default_rng(2).normal(size=(30,3));original=make_ply(tmp_path/'native.ply',xyz,True)
    transform_ply(tmp_path/'native.ply',tmp_path/'aligned.ply',truth);v=PlyData.read(tmp_path/'aligned.ply')['vertex'].data
    assert np.allclose(np.column_stack([v[n] for n in ('x','y','z')]),truth.apply(xyz),atol=1e-6)
    r=Rotation.from_quat(np.array([v[f'rot_{i}'][0] for i in (1,2,3,0)])).as_matrix()
    cov=r@np.diag(np.exp(2*np.array([v[f'scale_{i}'][0] for i in range(3)])))@r.T
    expected=truth.scale**2*truth.rotation@np.diag(np.exp(2*np.array([-3.,-2.,-1.])))@truth.rotation.T
    assert np.allclose(cov,expected,atol=1e-6)
    assert np.array_equal(v['opacity'],original['opacity'])
    assert np.array_equal(v['f_dc_1'],original['f_dc_1'])


def test_nonidentity_training_normalization(tmp_path,truth):
    native=Similarity(.3,Rotation.from_euler('z',57,degrees=True).as_matrix(),np.array([3.,1.,-4.]))
    write(tmp_path/'scene_transform.json',{'train_from_world':native.as_dict()})
    result=model_to_map(truth,tmp_path/'scene_transform.json');p=np.array([[4.,7.,1.]])
    assert np.allclose(result.apply(native.apply(p)),truth.apply(p))


def test_refuses_unknown_training_transform_and_overwrite(tmp_path,truth):
    write(tmp_path/'bad.json',{})
    with pytest.raises(ValueError):model_to_map(truth,tmp_path/'bad.json')
    make_ply(tmp_path/'native.ply',np.zeros((2,3)),True)
    with pytest.raises(FileExistsError):transform_ply(tmp_path/'native.ply',tmp_path/'native.ply',truth)


def test_zero_sh_padding_removed_but_nonzero_sh_refused(tmp_path,truth):
    from plyfile import PlyElement
    original=make_ply(tmp_path/'original.ply',np.ones((3,3)),True)
    extended=np.zeros(3,dtype=original.dtype.descr+[(f'f_rest_{i}','<f4') for i in range(48)])
    for name in original.dtype.names:extended[name]=original[name]
    PlyData([PlyElement.describe(extended,'vertex')],text=False).write(tmp_path/'padded.ply')
    transform_ply(tmp_path/'padded.ply',tmp_path/'sh0.ply',truth)
    assert not any(n.startswith('f_rest_') for n in PlyData.read(tmp_path/'sh0.ply')['vertex'].data.dtype.names)
    extended['f_rest_0']=.2
    PlyData([PlyElement.describe(extended,'vertex')],text=False).write(tmp_path/'nonzero.ply')
    with pytest.raises(ValueError,match='SH'):transform_ply(tmp_path/'nonzero.ply',tmp_path/'bad_export.ply',truth)
    assert not (tmp_path/'bad_export.ply').exists()


def test_colmap_transform_preserves_projective_camera_geometry(tmp_path,truth):
    poses={'a.jpg':np.eye(4)};points=np.array([[1.,2.,3.],[2,3,4],[3,4,5]])
    sparse=make_sparse(tmp_path/'sparse',points,poses);transform_colmap(sparse,tmp_path/'aligned',truth)
    camera=c2w_from_record(read_images(tmp_path/'aligned/images.bin')['a.jpg'])
    map_points=truth.apply(points);in_camera=(map_points-camera[:3,3])@camera[:3,:3]
    assert np.allclose(in_camera[:,:2]/in_camera[:,2:],points[:,:2]/points[:,2:])


def test_las_units_and_cache_integrity(tmp_path):
    import laspy
    las=laspy.LasData(laspy.LasHeader(point_format=1,version='1.2'));rng=np.random.default_rng(3);p=rng.uniform(100000,105000,(3000,3))
    las.x=p[:,0];las.y=p[:,1];las.z=p[:,2];las.write(tmp_path/'data.las')
    result,meta=read_lidar(tmp_path/'data.las',tmp_path/'cache',unit=.001,size=.01)
    assert np.all(result>=100) and np.all(result<=105)
    assert meta['inputs']['unit_to_m']==.001
    other,_=read_lidar(tmp_path/'data.las',tmp_path/'cache',unit=.001,size=.01)
    assert np.array_equal(result,other)
    with (tmp_path/'cache/lidar.npz').open('ab') as f:f.write(b'corrupt')
    repaired,_=read_lidar(tmp_path/'data.las',tmp_path/'cache',unit=.001,size=.01)
    assert np.array_equal(result,repaired)


def test_gaussian_not_accepted_as_measured_lidar(tmp_path):
    make_ply(tmp_path/'splats.ply',np.ones((600,3)),True)
    with pytest.raises(ValueError):list(lidar_chunks(tmp_path/'splats.ply'))


def test_resume_validates_completed_artifacts(tmp_path):
    path=tmp_path/'result';path.write_text('first')
    assert stage(tmp_path,'one',lambda:({'ok':True},[path]))['ok']
    path.write_text('changed')
    with pytest.raises(ValueError):stage(tmp_path,'one',lambda:({},[]))
    with pytest.raises(ValueError):disjoint(tmp_path,tmp_path/'input')
