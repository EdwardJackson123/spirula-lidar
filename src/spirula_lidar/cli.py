import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .artifacts import read,write,status
from . import __version__


def config_for(args):
    cfg={}
    if getattr(args,'config',None):
        path=Path(args.config).resolve();cfg=read(path)
        for key in ('images','lidar','output','sfm','reference_poses','person_model','model','scene_transform','landmarks'):
            if cfg.get(key) and not Path(cfg[key]).is_absolute():cfg[key]=str(path.parent/cfg[key])
        if cfg.get('spirula') and ('/' in cfg['spirula'] or '\\' in cfg['spirula']) and not Path(cfg['spirula']).is_absolute():cfg['spirula']=str(path.parent/cfg['spirula'])
    for key in ('images','lidar','output','sfm','spirula','person_model','reference_poses','model','scene_transform','landmarks','gpu_index','threads','lidar_unit_to_m','dry_run','ordered_images'):
        if hasattr(args,key) and getattr(args,key) is not None:cfg[key]=getattr(args,key)
    return cfg


def doctor(binary):
    from .engine import resolve_binary
    binary=resolve_binary(binary);env=os.environ.copy();env['SS_LANG']='en'
    modules={name:importlib.util.find_spec(name) is not None for name in ('numpy','scipy','PIL','laspy','plyfile','ultralytics','pycolmap','torch','cv2')}
    top=subprocess.check_output([binary,'--help'],env=env,text=True,stderr=subprocess.STDOUT,timeout=30)
    required={'sfm auto':['--camera-model','--sensor-gauge','--metric-gps'],
              'geometry':['--depth-units','--normal-format'],
              'train 3dgs':['--depth-supervision-weight','--normal-supervision-weight','--scene-center','--floater-suppression']}
    flags={}
    for command,names in required.items():
        out=subprocess.check_output([binary,*command.split(),'--help'],env=env,text=True,stderr=subprocess.STDOUT,timeout=30)
        flags[command]={name:name in out for name in names}
    result=dict(version=__version__,python=sys.version.split()[0],spirula=binary,spirula_version=top.splitlines()[0],modules=modules,required_cli=flags,nvidia_smi=bool(shutil.which('nvidia-smi')))
    result['ready']=bool(all(modules.values()) and all(all(f.values()) for f in flags.values()) and result['nvidia_smi'])
    print(json.dumps(result,indent=2));return 0 if result['ready'] else 1


def main(argv=None):
    parser=argparse.ArgumentParser(prog='spirula-lidar',description='Image-only Spirula training and checked metric LiDAR alignment')
    parser.add_argument('--version',action='version',version=__version__);sub=parser.add_subparsers(dest='command',required=True)
    for name in ('run','align'):
        p=sub.add_parser(name);p.add_argument('--config');p.add_argument('--lidar');p.add_argument('--output');p.add_argument('--sfm')
        p.add_argument('--reference-poses');p.add_argument('--landmarks');p.add_argument('--threads',type=int);p.add_argument('--lidar-unit-to-m',type=float)
        if name=='run':
            p.add_argument('--images');p.add_argument('--spirula');p.add_argument('--person-model');p.add_argument('--gpu-index',type=int)
            p.add_argument('--dry-run',action='store_true',default=None);p.add_argument('--ordered-images',action='store_true',default=None)
        else:p.add_argument('--model');p.add_argument('--scene-transform')
    d=sub.add_parser('doctor');d.add_argument('--spirula',default='spirula')
    s=sub.add_parser('status');s.add_argument('output')
    s=sub.add_parser('serve');s.add_argument('output');s.add_argument('--host',default='127.0.0.1');s.add_argument('--port',type=int,default=8765)
    args=parser.parse_args(argv);cfg=None
    try:
        if args.command=='doctor':return doctor(args.spirula)
        if args.command=='status':print(json.dumps(read(Path(args.output)/'status.json'),ensure_ascii=False,indent=2));return 0
        if args.command=='serve':
            from .monitor import serve
            serve(args.output,args.host,args.port);return 0
        cfg=config_for(args)
        # Set BLAS thread limits before importing the numerical worker modules.
        threads=str(cfg.get('threads',8));os.environ.update(OMP_NUM_THREADS=threads,OPENBLAS_NUM_THREADS=threads,MKL_NUM_THREADS=threads)
        from .pipeline import run,align_existing
        return run(cfg) if args.command=='run' else align_existing(cfg)
    except (Exception,KeyboardInterrupt) as error:
        root=Path(cfg['output']).resolve() if cfg and cfg.get('output') else None
        if root and (root/'job.json').exists():status(root,'error','stopped' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'needs_attention',error=type(error).__name__+': '+str(error))
        print(type(error).__name__+': '+str(error),file=sys.stderr);return 1
