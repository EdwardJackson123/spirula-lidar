"""Read-only capture IO. Point coordinates retain float64 precision."""
from pathlib import Path, PurePosixPath
import struct
import numpy as np
from scipy.spatial import cKDTree
from .artifacts import sha, read, write, fingerprint


def unpack(stream, fmt):
    size=struct.calcsize(fmt); data=stream.read(size)
    if len(data)!=size: raise ValueError("Truncated COLMAP record")
    return struct.unpack(fmt,data)


def safe_name(name):
    path=PurePosixPath(name)
    if not name or path.is_absolute() or any(p in ("..", ".") for p in path.parts) or "\\" in name or ":" in name:
        raise ValueError("Unsafe image name: "+name)
    return name


def read_images(path):
    result={}
    with Path(path).open("rb") as f:
        count,=unpack(f,"<Q")
        for _ in range(count):
            iid,*values=unpack(f,"<i7di"); raw=bytearray()
            while True:
                byte=f.read(1)
                if not byte: raise ValueError("Unterminated COLMAP image name")
                if byte==b"\0": break
                raw.extend(byte)
                if len(raw)>65536: raise ValueError("Invalid image name length")
            name=safe_name(raw.decode("utf8")); n,=unpack(f,"<Q")
            observations=f.read(n*24)
            if len(observations)!=n*24: raise ValueError("Truncated COLMAP observations")
            if name in result: raise ValueError("Duplicate COLMAP image name")
            result[name]=dict(id=iid,q=np.array(values[:4]),t=np.array(values[4:7]),camera_id=values[7],observations=observations)
        if f.read(1): raise ValueError("Trailing COLMAP image bytes")
    return result


def read_sfm_points(path, max_points=120000, min_track=3, max_error=1.):
    rng=np.random.default_rng(7103); points=np.empty((max_points,3)); errors=[]; seen=0; total=0
    with Path(path).open("rb") as f:
        total,=unpack(f,"<Q")
        for _ in range(total):
            row=unpack(f,"<QdddBBBdQ"); track=f.read(row[-1]*8)
            if len(track)!=row[-1]*8: raise ValueError("Truncated COLMAP track")
            if row[-1]<min_track or not np.isfinite(row[1:4]).all() or not np.isfinite(row[7]) or row[7]>max_error: continue
            slot=seen if seen<max_points else int(rng.integers(seen+1))
            if slot<max_points: points[slot]=row[1:4]
            errors.append(row[7]); seen+=1
        if f.read(1): raise ValueError("Trailing COLMAP point bytes")
    if seen<500: raise ValueError("Too few reliable image points for alignment")
    return points[:min(seen,max_points)],dict(total_points=total,qualified_points=seen,sampled_points=min(seen,max_points),filtered_point_error_median_px=float(np.median(errors)))


def voxel(points, size, origin=None):
    points=np.asarray(points,float); points=points[np.isfinite(points).all(1)]
    if not len(points): return points
    origin=points.min(0) if origin is None else origin
    keys=np.floor((points-origin)/size).astype(np.int64)
    _,ids=np.unique(keys,axis=0,return_index=True)
    return points[np.sort(ids)]


def source_cloud(points):
    extent=float(np.linalg.norm(np.quantile(points,.99,axis=0)-np.quantile(points,.01,axis=0)))
    if extent<=1e-8: raise ValueError("Degenerate source extent")
    points=voxel(points,extent/250)
    distance=cKDTree(points).query(points,k=9,workers=4)[0][:,-1]
    points=points[distance<=np.quantile(distance,.995)]
    return points


def spatial_holdout(points, fraction=.2):
    x=np.asarray(points,float); span=np.linalg.norm(np.quantile(x,.99,axis=0)-np.quantile(x,.01,axis=0))
    if span<=1e-9: raise ValueError("Degenerate point cloud")
    cells=np.floor((x-np.min(x,axis=0))/(span/20)).astype(np.int64)
    hashed=(cells[:,0]*73856093)^(cells[:,1]*19349663)^(cells[:,2]*83492791)
    mask=(hashed%1000)<round(fraction*1000)
    if mask.sum()<100 or (~mask).sum()<500: raise ValueError("Insufficient spatial fit/holdout blocks")
    return mask


def lidar_chunks(path):
    path=Path(path)
    if path.suffix.lower() in (".las",".laz"):
        import laspy
        with laspy.open(path) as stream:
            for p in stream.chunk_iterator(750000): yield np.column_stack((p.x,p.y,p.z))
    elif path.suffix.lower()==".ply":
        from plyfile import PlyData
        v=PlyData.read(path,mmap="r")["vertex"].data
        if not {"x","y","z"}.issubset(v.dtype.names): raise ValueError("PLY has no xyz")
        if any(n.startswith(("f_dc_","scale_","rot_")) for n in v.dtype.names): raise ValueError("Expected measured LiDAR, received a Gaussian model")
        for start in range(0,len(v),750000): yield np.column_stack([v[n][start:start+750000] for n in ("x","y","z")])
    else: raise ValueError("LiDAR input must be LAS, LAZ or point-cloud PLY")


def read_lidar(path, output, unit=1., size=.08, max_points=600000, progress=None):
    if not np.isfinite(unit) or unit<=0 or not np.isfinite(size) or size<=0 or max_points<1000: raise ValueError("Invalid LiDAR units/voxel/budget")
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    signature=dict(sha256=sha(path),unit_to_m=unit,initial_voxel_m=size,max_points=max_points,reader_version=1)
    marker=output/"lidar.json"; cache=output/"lidar.npz"
    if marker.exists() and cache.exists():
        meta=read(marker)
        if meta["inputs"]==signature and sha(cache)==meta["cache_sha256"]:
            with np.load(cache) as d: return d["points"],meta
    points=np.empty((0,3)); count=0; origin=None
    for raw in lidar_chunks(path):
        count+=len(raw); p=np.asarray(raw,float)*unit; p=p[np.isfinite(p).all(1)]
        if not len(p): continue
        if origin is None: origin=p.min(0)
        p=voxel(p,size,origin); points=voxel(np.concatenate((points,p)),size,origin)
        while len(points)>max_points:
            size*=1.25; points=voxel(points,size,origin)
        if progress: progress("read_lidar",points_read=count,retained_points=len(points),effective_voxel_m=size)
    if len(points)<500: raise ValueError("Insufficient finite LiDAR points")
    temp=cache.with_name("lidar.tmp.npz");np.savez_compressed(temp,points=points);temp.replace(cache)
    meta=dict(inputs=signature,cache_sha256=sha(cache),raw_points=count,retained_points=len(points),effective_voxel_m=size)
    write(marker,meta);return points,meta


def normals(points, knn=24, threads=4):
    x=np.asarray(points,float); tree=cKDTree(x); result=np.empty_like(x)
    for start in range(0,len(x),20000):
        ids=tree.query(x[start:start+20000],k=min(knn,len(x)),workers=threads)[1]
        neighbors=x[ids]; neighbors-=neighbors.mean(1,keepdims=True)
        covariance=np.einsum("nki,nkj->nij",neighbors,neighbors)
        result[start:start+len(ids)]=np.linalg.eigh(covariance)[1][:,:,0]
    return result
