import os
from pathlib import Path
import shutil
import sys
import numpy as np
from PIL import Image
from . import __version__
from .artifacts import read,write,sha,fingerprint,disjoint,inside,lock,status,stage
from . import engine
from .clouds import read_images,read_sfm_points,source_cloud,read_lidar,safe_name
from .alignment import align_points
from .sim3 import Similarity
from .gaussians import model_to_map,transform_ply,transform_colmap


def validate(config,training=True):
    cfg=dict(config)
    for key in ('images','lidar','output') if training else ('sfm','lidar','output'):
        if not cfg.get(key):raise ValueError('Missing configuration: '+key)
        cfg[key]=str(Path(cfg[key]).expanduser().resolve())
    for key in ('sfm','reference_poses','person_model','model','scene_transform','landmarks'):
        if cfg.get(key) and (key!='person_model' or Path(cfg[key]).is_file()):cfg[key]=str(Path(cfg[key]).expanduser().resolve())
    output=Path(cfg['output']);inputs=[cfg.get(k) for k in ('images','lidar','sfm','reference_poses','model','scene_transform','landmarks')]
    disjoint(output,*inputs)
    for key in ('images','lidar','sfm','reference_poses','model','scene_transform','landmarks'):
        if cfg.get(key) and not Path(cfg[key]).exists():raise FileNotFoundError(cfg[key])
    unit=cfg.get('lidar_unit_to_m',1.)
    if not isinstance(unit,(int,float)) or not np.isfinite(unit) or unit<=0:raise ValueError('Declare positive LiDAR unit conversion')
    if type(cfg.get('threads',8)) is not int or not 1<=cfg.get('threads',8)<=16:raise ValueError('threads must be 1..16')
    cfg.setdefault('threads',8);cfg.setdefault('lidar_unit_to_m',1.)
    resources=cfg.get('resources',{})
    for key,default,maximum in [('gpu_gib',18,18),('ram_gib',64,64)]:
        v=resources.get(key,default)
        if not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<v<=maximum:raise ValueError('Resource budget invalid: '+key)
    fraction=resources.get('free_reserve_fraction',.2)
    if not isinstance(fraction,(int,float)) or not np.isfinite(fraction) or not .2<=fraction<.8:raise ValueError('GPU reserve fraction must be .2.. .8')
    alignment=cfg.get('alignment',{})
    for key in ('max_scale_hypotheses','coarse_per_mode','refine_per_mode','yaw_step_degrees'):
        if key in alignment and (type(alignment[key]) is not int or alignment[key]<=0):raise ValueError('Invalid alignment integer: '+key)
    if alignment.get('yaw_step_degrees',4)>30:raise ValueError('yaw_step_degrees cannot exceed 30')
    if cfg.get('model') and not cfg.get('scene_transform'):raise ValueError('Model export requires its Spirula scene_transform.json')
    if training:
        cfg['spirula']=engine.resolve_binary(cfg.get('spirula','spirula'))
        cfg.setdefault('person_model','yolo11x-seg.pt')
    return cfg


def input_files(cfg,include_images=True):
    from importlib.metadata import version,PackageNotFoundError
    result={};records=[];stems=set()
    if include_images:
        root=Path(cfg['images'])
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in ('.jpg','.jpeg','.png','.webp'):continue
            name=safe_name(path.relative_to(root).as_posix());stem=Path(name).with_suffix('').as_posix()
            if stem in stems:raise ValueError('Ambiguous mask stems: '+name)
            stems.add(stem)
            with Image.open(path) as im:
                if im.getexif().get(274,1)!=1:raise ValueError('Resolve EXIF orientation consistently before this run: '+name)
                w,h=im.size
            records.append(dict(name=name,width=w,height=h,sha256=sha(path)))
        if len(records)<12:raise ValueError('At least 12 overlapping images required')
        result['images']=records
    for key in ('lidar','sfm','reference_poses','model','scene_transform','landmarks'):
        if not cfg.get(key):continue
        path=Path(cfg[key]);paths=[path]
        if path.is_dir():paths=[path/n for n in ('cameras.bin','images.bin','points3D.bin')]
        result[key]=[dict(path=str(p),sha256=sha(p)) for p in paths]
    if cfg.get('spirula'):result['spirula_sha256']=sha(cfg['spirula'])
    if cfg.get('person_model') and Path(cfg['person_model']).is_file():result['person_model_sha256']=sha(cfg['person_model'])
    result['code_sha256']=fingerprint({p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    result['dependency_versions']={}
    for name in ('numpy','scipy','Pillow','laspy','plyfile','pycolmap','ultralytics','torch'):
        try:result['dependency_versions'][name]=version(name)
        except PackageNotFoundError:pass
    return result,records


def init_job(cfg,inputs):
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    stamp=fingerprint(dict(config=cfg,inputs=inputs))
    if (root/'job.json').exists():
        if read(root/'job.json')['fingerprint']!=stamp:raise ValueError('Input/config/code changed: choose a new output directory')
    else:
        if any(root.iterdir()):
            unexpected=[p.name for p in root.iterdir() if p.name!='.lock']
            if unexpected:raise ValueError('New output directory is not empty: '+', '.join(unexpected[:5]))
        write(root/'job.json',dict(version=__version__,fingerprint=stamp,config=cfg,inputs=inputs))


def alignment_stage(cfg,sparse,root):
    work=root/'alignment';work.mkdir(exist_ok=True)
    def progress(name,**values):
        if (root/'STOP').exists():raise InterruptedError('STOP requested')
        status(root,name,**values)
    points,audit=read_sfm_points(Path(sparse)/'points3D.bin');source=source_cloud(points)
    settings={**cfg.get('alignment',{}),'threads':cfg['threads']}
    target,meta=read_lidar(cfg['lidar'],root/'cache/lidar',unit=cfg['lidar_unit_to_m'],size=settings.get('lidar_voxel_m',.08),progress=progress)
    settings['lidar_voxel_m']=meta['effective_voxel_m']
    write(work/'input_provenance.json',dict(sfm_points_sha256=sha(Path(sparse)/'points3D.bin'),lidar=meta,sampling=audit,
                reference_poses=cfg.get('reference_poses'),settings=settings))
    if (work/'alignment.json').exists():
        # A completed report can only be reused under the enclosing immutable job fingerprint.
        report=read(work/'alignment.json')
        if report.get('sealed_solution_sha256') and sha(work/'sealed_solution.json')!=report['sealed_solution_sha256']:raise ValueError('Sealed alignment checksum changed')
    elif (work/'sealed_solution.json').exists():
        raise ValueError('Interrupted after sealing: preserve evidence and use a fresh output directory')
    else:
        report=align_points(source,target,work,settings,source_images=read_images(Path(sparse)/'images.bin'),reference_poses=cfg.get('reference_poses'),landmarks=cfg.get('landmarks'),progress=progress)
    paths=[work/'alignment.json',work/'input_provenance.json']
    for name in ('sealed_solution.json','preview.html','proposals.json'):
        if (work/name).exists():paths.append(work/name)
    return report,paths


def export_stage(cfg,sparse,report,root,model,scene_transform):
    if not report.get('passed'):raise ValueError('Unaccepted alignment cannot be exported as metric')
    if read(root/'alignment/alignment.json')!=report:raise ValueError('Export report does not match the persisted alignment evidence')
    alignment=Similarity.from_dict(report['similarity']);transform=model_to_map(alignment,scene_transform)
    import tempfile
    destination=inside(root,root/'output')
    if destination.exists():raise FileExistsError('Existing final output is never overwritten')
    temporary=Path(tempfile.mkdtemp(prefix='.export-',dir=root))
    model_info=transform_ply(model,temporary/'scene.ply',transform)
    model_info['path']=str(destination/'scene.ply')
    colmap=transform_colmap(sparse,temporary/'sparse/0',alignment)
    manifest=dict(version=__version__,model=model_info,frame='lidar',unit='metres',map_from_sfm=alignment.as_dict(),map_from_train=transform.as_dict(),
                  alignment_report_sha256=sha(root/'alignment/alignment.json'),training_transform_sha256=sha(scene_transform),
                  native_model_sha256=sha(model),uses_lidar_in_training=False,uses_lidar_for_alignment=True,
                  certification='not_certified',absolute_survey_accuracy_verified=False,
                  limitation='Uniform alignment does not correct local reconstruction distortion or certify collision geometry')
    write(temporary/'manifest.json',manifest);temporary.replace(destination)
    return manifest,[destination/'scene.ply',destination/'manifest.json',*[destination/p.relative_to(temporary) for p in colmap]]


def align_existing(config):
    cfg=validate(config,False);root=Path(cfg['output'])
    with lock(root/'.lock'):
        inputs,_=input_files(cfg,False);init_job(cfg,inputs)
        report=stage(root,'alignment',lambda:alignment_stage(cfg,Path(cfg['sfm']),root))
        if report['passed'] and cfg.get('model'):
            stage(root,'export',lambda:export_stage(cfg,cfg['sfm'],report,root,cfg['model'],cfg['scene_transform']))
        state='completed' if report['passed'] else 'needs_review';status(root,'complete',state,alignment_passed=report['passed'])
        return 0 if report['passed'] else 2


def audit_sfm(sparse,images):
    import pycolmap
    reconstruction=pycolmap.Reconstruction(str(sparse));errors=[]
    for image in reconstruction.images.values():
        observations=[p for p in image.points2D if p.has_point3D() and p.point3D_id in reconstruction.points3D][::8]
        if not observations:continue
        xyz=np.array([reconstruction.points3D[p.point3D_id].xyz for p in observations]);tf=image.cam_from_world().matrix()
        camera=xyz@tf[:,:3].T+tf[:,3];inside=camera[:,2]>1e-6
        uv=reconstruction.cameras[image.camera_id].img_from_cam(camera[inside])
        xy=np.array([p.xy for p in observations])[inside];errors.extend(np.linalg.norm(uv-xy,axis=1))
    if len(errors)<100:raise ValueError('Insufficient valid SfM observations')
    audit=dict(registered_images=reconstruction.num_reg_images(),coverage=reconstruction.num_reg_images()/len(images),points=reconstruction.num_points3D(),
               reprojection_median_px=float(np.median(errors)),reprojection_p95_px=float(np.percentile(errors,95)),sampled_observations=len(errors))
    audit['passed']=bool(audit['coverage']>=.95 and audit['reprojection_median_px']<=1.5 and audit['reprojection_p95_px']<=3 and audit['points']>=1000)
    return audit


class Job:
    def __init__(self,cfg,records,lock_fd):
        self.cfg=cfg;self.root=Path(cfg['output']);self.data=self.root/'dataset';self.records=records;self.lock_fd=lock_fd
        self.env=os.environ.copy();self.env.update(SS_LANG='en',OMP_NUM_THREADS=str(cfg['threads']),OPENBLAS_NUM_THREADS=str(cfg['threads']))
        for key,name in [('XDG_CACHE_HOME','xdg'),('TORCH_HOME','torch'),('YOLO_CONFIG_DIR','yolo'),('MPLCONFIGDIR','matplotlib')]:
            path=self.root/'cache'/name;path.mkdir(parents=True,exist_ok=True);self.env[key]=str(path)
        self.index=cfg.get('gpu_index',0);self.memory=engine.gpu_info(self.index);self.device=self.memory['device']

    def command(self,name,args):engine.command(self.root,name,args,self.env,self.lock_fd,self.index,self.cfg.get('resources',{}))

    def prepare(self):
        paths=[]
        for row in self.records:
            dest=inside(self.root,self.data/'images'/row['name']);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(self.cfg['images'])/row['name'],dest)
            if sha(dest)!=row['sha256']:raise ValueError('Image copy checksum mismatch')
            paths.append(dest)
        write(self.root/'images.json',self.records)
        return self.records,[self.root/'images.json',*paths]

    def masks(self):
        import cv2,torch
        from ultralytics import YOLO
        torch.set_num_threads(self.cfg['threads']);os.environ.update({k:v for k,v in self.env.items() if k.endswith(('_HOME','_DIR'))})
        model_path=Path(self.cfg['person_model'])
        if not model_path.is_absolute():
            # Ultralytics downloads known model names beside this private cache.
            cache=self.root/'cache/person';cache.mkdir(parents=True,exist_ok=True);model_path=cache/model_path.name
        model=YOLO(str(model_path));paths=[]
        for i,row in enumerate(self.records):
            if (self.root/'STOP').exists():raise InterruptedError('STOP requested')
            result=model.predict(str(self.data/'images'/row['name']),imgsz=min(max(row['width'],row['height']),1600),conf=.15,classes=[0],retina_masks=True,device=self.index,verbose=False,save=False)[0]
            mask=np.zeros((row['height'],row['width']),np.uint8)
            if result.masks is not None:
                mask=(result.masks.data.amax(0).cpu().numpy()>.5).astype(np.uint8);mask=cv2.resize(mask,(row['width'],row['height']),interpolation=cv2.INTER_NEAREST)
            mask=cv2.dilate(mask,np.ones((9,9),np.uint8));path=inside(self.root,(self.data/'masks'/row['name']).with_suffix('.png'));path.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray((1-mask)*255).save(path);paths.append(path);status(self.root,'masks',processed=i+1,total=len(self.records))
        del model;torch.cuda.empty_cache();write(self.root/'person_model.json',dict(path=str(model_path),sha256=sha(model_path),mask_semantics='white=keep, black=ignore',classes=['person']))
        return dict(images=len(paths)),[*paths,self.root/'person_model.json']

    def sfm(self):
        output=self.root/'sfm_work'
        if self.cfg.get('sfm'):sparse=Path(self.cfg['sfm'])
        else:
            self.command('sfm',engine.sfm_command(self.cfg['spirula'],self.data,output,self.cfg,self.device))
            candidates=[p.parent for p in output.rglob('images.bin') if (p.parent/'points3D.bin').exists()]
            if not candidates:raise ValueError('Spirula did not produce a COLMAP model')
            sparse=max(candidates,key=lambda p:len(read_images(p/'images.bin')))
        audit=audit_sfm(sparse,self.records);write(self.root/'sfm_audit.json',audit)
        if not audit['passed']:raise ValueError('SfM coverage/reprojection checks failed; see sfm_audit.json')
        dest=self.data/'sparse/0';dest.mkdir(parents=True,exist_ok=True);paths=[]
        for name in ('cameras.bin','images.bin','points3D.bin'):
            shutil.copyfile(sparse/name,dest/name);paths.append(dest/name)
        return audit,[self.root/'sfm_audit.json',*paths]

    def geometry(self):
        maximum=max(max(r['width'],r['height']) for r in self.records)
        self.command('geometry',engine.geometry_command(self.cfg['spirula'],self.data,maximum,self.device));paths=[]
        sizes={r['name']:(r['width'],r['height']) for r in self.records}
        for name in read_images(self.data/'sparse/0/images.bin'):
            for kind in ('depths','normals'):
                path=(self.data/kind/name).with_suffix('.png')
                with Image.open(path) as image:
                    values=np.array(image)
                    if image.size!=sizes[name] or not values.size or np.mean(values>0)<.1:raise ValueError('Invalid predicted geometry: '+str(path))
                paths.append(path)
        return dict(model='moge2-vitl',depth_units='relative',lidar_supervision=False),paths

    def budget(self,audit):
        selected=engine.plan_budget(audit,self.records,engine.gpu_info(self.index),self.cfg);write(self.root/'budget.json',selected)
        return selected,[self.root/'budget.json']

    def train(self,budget):
        output=self.root/'training/final';resume=bool(list(output.glob('step-*.ckpt')))
        self.command('train',engine.training_command(self.cfg['spirula'],self.data,self.root,budget,self.device,resume))
        actual=read(output/'config.json')
        for k,v in dict(sh_degree=0,load_masks=True,load_depths=True,load_normals=True,depth_supervision_weight=.01,normal_supervision_weight=.01,floater_suppression='mild',train_resolution_divisor=1,num_iterations=budget['iterations'],cap_max=budget['gaussians']).items():
            if actual.get(k)!=v:raise ValueError('Actual training configuration mismatch: '+k)
        model=output/f'step-{budget["iterations"]:09d}.ckpt/splat.ply'
        return dict(model=str(model),scene_transform=str(output/'scene_transform.json')),[model,output/'scene_transform.json',output/'config.json',output/'metrics.json']


def run(config):
    cfg=validate(config);dry_run=cfg.pop('dry_run',False);root=Path(cfg['output'])
    with lock(root/'.lock') as fd:
        inputs,records=input_files(cfg);init_job(cfg,inputs)
        if dry_run:
            write(root/'plan.json',dict(stages=['prepare','masks','sfm','alignment','geometry','budget','train','export'],defaults=dict(sh_degree=0,depth_weight=.01,normal_weight=.01,floater_suppression='mild'),note='Dry run: no model inference or training executed'))
            status(root,'planned','planned');return 0
        with lock(root.parent/f'.spirula-gpu-{cfg.get("gpu_index",0)}.lock') as gpu_fd:
            job=Job(cfg,records,(fd,gpu_fd))
            stage(root,'prepare',job.prepare);stage(root,'masks',job.masks);audit=stage(root,'sfm',job.sfm)
            alignment=stage(root,'alignment',lambda:alignment_stage(cfg,job.data/'sparse/0',root))
            if not alignment['passed']:
                status(root,'alignment','needs_review',reason=alignment['reason']);return 2
            stage(root,'geometry',job.geometry);budget=stage(root,'budget',lambda:job.budget(audit));trained=stage(root,'train',lambda:job.train(budget))
            stage(root,'export',lambda:export_stage(cfg,job.data/'sparse/0',alignment,root,trained['model'],trained['scene_transform']))
            status(root,'complete','completed',model=str(root/'output/scene.ply'));return 0
