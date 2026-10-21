import math
from pathlib import Path
import shutil
import struct
import numpy as np
from scipy.spatial.transform import Rotation
from .sim3 import Similarity, c2w_from_record, transform_pose
from .clouds import read_images, unpack
from .artifacts import read, sha


def model_to_map(alignment, scene_transform):
    data=read(scene_transform)
    if "train_from_world" not in data: raise ValueError("Spirula scene transform lacks train_from_world")
    return alignment.compose(Similarity.from_dict(data["train_from_world"]).inverse())


def transform_ply(source, destination, transform):
    from plyfile import PlyData, PlyElement
    source,destination=Path(source),Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    v=PlyData.read(source,mmap="r")["vertex"].data
    required={"x","y","z","opacity",*[f"f_dc_{i}" for i in range(3)],*[f"scale_{i}" for i in range(3)],*[f"rot_{i}" for i in range(4)]}
    if not len(v) or not required.issubset(v.dtype.names): raise ValueError("Invalid Gaussian PLY")
    rest=[n for n in v.dtype.names if n.startswith("f_rest_")]
    for name in rest:
        if np.any(v[name]!=0): raise ValueError("Nonzero SH>0 requires SH coefficient rotation; only SH0 is supported")
    names=[n for n in v.dtype.names if n not in rest]
    dtype=np.dtype([(n,v.dtype.fields[n][0]) for n in names])
    header=("ply\nformat binary_little_endian 1.0\nelement vertex "+str(len(v))+"\n")
    types={"f4":"float","f8":"double","u1":"uchar","i4":"int","u4":"uint"}
    for n in names:
        key=dtype.fields[n][0].kind+str(dtype.fields[n][0].itemsize)
        if key not in types: raise ValueError("Unsupported Gaussian property type")
        header+=f"property {types[key]} {n}\n"
    destination.parent.mkdir(parents=True,exist_ok=True); tmp=destination.with_name(destination.name+".partial")
    try:
        with tmp.open("wb") as stream:
            stream.write((header+"end_header\n").encode("ascii"))
            for start in range(0,len(v),100000):
                original=v[start:start+100000]; chunk=np.empty(len(original),dtype=dtype.newbyteorder("<"))
                for n in names:
                    if not np.isfinite(original[n]).all(): raise ValueError("Nonfinite Gaussian attribute: "+n)
                    chunk[n]=original[n]
                xyz=np.column_stack([chunk[n] for n in ("x","y","z")]);xyz=transform.apply(xyz)
                for i,n in enumerate(("x","y","z")): chunk[n]=xyz[:,i]
                for i in range(3): chunk[f"scale_{i}"]+=math.log(transform.scale)
                q=np.column_stack([chunk[f"rot_{i}"] for i in range(4)]).astype(float)
                if np.any(np.linalg.norm(q,axis=1)<1e-10): raise ValueError("Zero Gaussian quaternion")
                rotation=Rotation.from_matrix(transform.rotation)*Rotation.from_quat(q[:,[1,2,3,0]])
                q=rotation.as_quat()[:,[3,0,1,2]]
                for i in range(4): chunk[f"rot_{i}"]=q[:,i]
                if {"nx","ny","nz"}.issubset(names):
                    n=np.column_stack([chunk[k] for k in ("nx","ny","nz")])@transform.rotation.T
                    for i,k in enumerate(("nx","ny","nz")): chunk[k]=n[:,i]
                for n in names:
                    if not np.isfinite(chunk[n]).all(): raise ValueError("Transformed PLY overflow: "+n)
                chunk.tofile(stream)
        tmp.replace(destination)
    except BaseException:
        if tmp.exists(): tmp.unlink()
        raise
    return dict(path=str(destination),sha256=sha(destination),gaussians=len(v),sh_degree=0,bytes=destination.stat().st_size)


def transform_colmap(sparse, output, transform):
    sparse,output=Path(sparse),Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copyfile(sparse/"cameras.bin",output/"cameras.bin")
    images=read_images(sparse/"images.bin")
    with (output/"images.bin").open("wb") as f:
        f.write(struct.pack("<Q",len(images)))
        for name,record in images.items():
            c2w=transform_pose(c2w_from_record(record),transform);r=c2w[:3,:3].T;t=-r@c2w[:3,3]
            q=Rotation.from_matrix(r).as_quat()[[3,0,1,2]]
            f.write(struct.pack("<i7di",record["id"],*q,*t,record["camera_id"]))
            f.write(name.encode()+b"\0");f.write(struct.pack("<Q",len(record["observations"])//24));f.write(record["observations"])
    with (sparse/"points3D.bin").open("rb") as src,(output/"points3D.bin").open("wb") as dst:
        count,=unpack(src,"<Q");dst.write(struct.pack("<Q",count))
        for _ in range(count):
            row=unpack(src,"<QdddBBBdQ");track=src.read(row[-1]*8)
            if len(track)!=row[-1]*8: raise ValueError("Truncated COLMAP track")
            xyz=transform.apply(np.array(row[1:4]));dst.write(struct.pack("<QdddBBBdQ",row[0],*xyz,*row[4:]));dst.write(track)
    return [output/n for n in ("cameras.bin","images.bin","points3D.bin")]
