from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "scale", float(self.scale))
        object.__setattr__(self, "rotation", np.asarray(self.rotation, dtype=float))
        object.__setattr__(self, "translation", np.asarray(self.translation, dtype=float))
        r, t = self.rotation, self.translation
        if not np.isfinite(self.scale) or self.scale <= 0 or r.shape != (3,3) or t.shape != (3,):
            raise ValueError("Invalid similarity dimensions/scale")
        if not np.isfinite(r).all() or not np.isfinite(t).all() or not np.allclose(r.T@r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1, atol=1e-6):
            raise ValueError("Similarity rotation must be finite, proper and orthonormal")

    @classmethod
    def identity(cls): return cls(1, np.eye(3), np.zeros(3))

    @classmethod
    def from_dict(cls, value):
        r = value.get("rotation", np.eye(3))
        if isinstance(r, dict): r = r["matrix_3x3"]
        result = cls(value["scale"], r, value["translation"])
        if "matrix_4x4" in value and not np.allclose(result.matrix(), value["matrix_4x4"], atol=1e-6):
            raise ValueError("Inconsistent placement matrix")
        return result

    def as_dict(self): return dict(scale=self.scale, rotation=self.rotation.tolist(), translation=self.translation.tolist())
    def apply(self, points): return self.scale*np.asarray(points)@self.rotation.T+self.translation
    def matrix(self):
        m=np.eye(4); m[:3,:3]=self.scale*self.rotation; m[:3,3]=self.translation; return m
    def inverse(self): return Similarity(1/self.scale, self.rotation.T, -self.rotation.T@self.translation/self.scale)
    def compose(self, other):
        """Return self(other(x))."""
        return Similarity(self.scale*other.scale, self.rotation@other.rotation, self.scale*self.rotation@other.translation+self.translation)


def umeyama(source, target):
    x, y = np.asarray(source, float), np.asarray(target, float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3 or not np.isfinite([x,y]).all():
        raise ValueError("Need finite matched Nx3 points")
    a, b=x-x.mean(0), y-y.mean(0)
    singular=np.linalg.svd(a, compute_uv=False)
    if singular[1] < max(singular[0]*1e-6, 1e-10): raise ValueError("Collinear/degenerate point correspondences")
    u, s, vt=np.linalg.svd(b.T@a/len(x)); signs=np.ones(3); signs[-1]=np.linalg.det(u@vt)
    r=u@np.diag(signs)@vt; scale=float(s@signs/np.mean(np.sum(a*a, axis=1)))
    return Similarity(scale, r, y.mean(0)-scale*r@x.mean(0))


def robust_similarity(source, target, tolerance, seed=19, trials=1000):
    x,y=np.asarray(source,float),np.asarray(target,float); rng=np.random.default_rng(seed); best=None
    if len(x)<6: raise ValueError("At least six matched landmarks required")
    for _ in range(trials):
        ids=rng.choice(len(x),3,replace=False)
        try: transform=umeyama(x[ids],y[ids])
        except ValueError: continue
        d=np.linalg.norm(transform.apply(x)-y,axis=1); keep=d<tolerance
        key=(int(keep.sum()),-float(np.minimum(d,tolerance).sum()))
        if best is None or key>best[0]: best=(key,keep)
    if best is None or best[1].sum()<6: raise ValueError("Insufficient landmark consensus")
    keep=best[1]
    for _ in range(5):
        transform=umeyama(x[keep],y[keep]); update=np.linalg.norm(transform.apply(x)-y,axis=1)<tolerance
        if update.sum()<6 or np.array_equal(update,keep): break
        keep=update
    return umeyama(x[keep],y[keep]),keep


def c2w_from_record(record):
    r=Rotation.from_quat(np.asarray(record["q"])[[1,2,3,0]]).as_matrix()
    m=np.eye(4);m[:3,:3]=r.T;m[:3,3]=-r.T@record["t"];return m


def transform_pose(c2w, transform):
    pose=np.asarray(c2w,float).copy()
    pose[:3,:3]=transform.rotation@pose[:3,:3]
    pose[:3,3]=transform.apply(pose[:3,3]);return pose
