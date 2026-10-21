import contextlib
import hashlib
import json
import os
from pathlib import Path
import time


def encode(value):
    import numpy as np
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=encode), encoding="utf-8")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""): h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=encode, allow_nan=False).encode()).hexdigest()


def disjoint(output, *inputs):
    output = Path(output).resolve()
    if output == Path(output.anchor): raise ValueError("Output cannot be a filesystem root")
    for item in inputs:
        if item is None: continue
        item = Path(item).resolve()
        if output == item or output in item.parents or item in output.parents:
            raise ValueError(f"Input and output trees overlap: {item}")


def inside(root, path):
    path = Path(path).resolve(); root = Path(root).resolve()
    if path == root or root not in path.parents: raise ValueError("Writable artifact escapes output: " + str(path))
    return path


@contextlib.contextmanager
def lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        if os.name == "posix":
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            stream.seek(0)
            if not stream.read(1): stream.write(b"0"); stream.flush()
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        yield stream.fileno()
    finally:
        stream.close()


def status(root, stage, state="running", **fields):
    write(Path(root)/"status.json", dict(status=state, stage=stage, pid=os.getpid(), updated_at=time.time(), **fields))


def stage(root, name, action):
    root = Path(root); marker = inside(root, root/"stages"/(name+".json"))
    if (root/"STOP").exists(): raise InterruptedError("STOP requested")
    if marker.exists():
        saved = read(marker)
        for item in saved["artifacts"]:
            if not Path(item["path"]).is_file() or sha(item["path"]) != item["sha256"]:
                raise ValueError("Completed artifact changed; use a new output directory: " + item["path"])
        return saved["value"]
    status(root, name)
    value, paths = action()
    write(marker, dict(value=value, artifacts=[dict(path=str(Path(p).resolve()), sha256=sha(p)) for p in paths]))
    return value
