from pathlib import Path
import struct
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from plyfile import PlyData,PlyElement
from spirula_lidar.sim3 import Similarity


@pytest.fixture
def truth():return Similarity(2.37,Rotation.from_euler('zyx',[41,7,-4],degrees=True).as_matrix(),np.array([4.,-8.,2.]))


def make_ply(path,xyz,gaussian=False):
    names=['x','y','z']
    if gaussian:names+=['nx','ny','nz','f_dc_0','f_dc_1','f_dc_2','opacity','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
    data=np.zeros(len(xyz),dtype=[(n,'<f4') for n in names])
    for i,n in enumerate(('x','y','z')):data[n]=xyz[:,i]
    if gaussian:
        data['rot_0']=1;data['opacity']=.27;data['f_dc_1']=.49;data['scale_0']=-3;data['scale_1']=-2;data['scale_2']=-1
    Path(path).parent.mkdir(parents=True,exist_ok=True);PlyData([PlyElement.describe(data,'vertex')],text=False,byte_order='<').write(path)
    return data


def make_sparse(path,points,poses):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    with (path/'cameras.bin').open('wb') as f:f.write(struct.pack('<QiiQQ4d',1,1,1,16,12,12.,12.,8.,6.))
    with (path/'images.bin').open('wb') as f:
        f.write(struct.pack('<Q',len(poses)))
        for i,(name,pose) in enumerate(poses.items(),1):
            r=pose[:3,:3].T;t=-r@pose[:3,3];q=Rotation.from_matrix(r).as_quat()[[3,0,1,2]]
            f.write(struct.pack('<i7di',i,*q,*t,1));f.write(name.encode()+b'\0');f.write(struct.pack('<Q',0))
    with (path/'points3D.bin').open('wb') as f:
        f.write(struct.pack('<Q',len(points)))
        for i,p in enumerate(points,1):f.write(struct.pack('<QdddBBBdQ6i',i,*p,100,120,140,.1,3,1,0,2,0,3,0))
    return path
