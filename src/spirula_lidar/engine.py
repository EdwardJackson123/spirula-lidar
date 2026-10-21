"""Spirula is an external native executable; no training implementation is forked."""
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from .artifacts import write,status


def resolve_binary(value='spirula'):
    result=shutil.which(str(value))
    if not result:raise FileNotFoundError('Spirula executable not found: '+str(value))
    return str(Path(result).resolve())


def gpu_info(index):
    raw=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=memory.free,memory.total,uuid','--format=csv,noheader,nounits'],text=True,timeout=15)
    free,total,uuid=raw.strip().split(',')
    return dict(free_gib=float(free)/1024,total_gib=float(total)/1024,device='uuid:'+uuid.strip().removeprefix('GPU-').replace('-',''))


def sfm_command(binary,data,output,config,device):
    args=[binary,'sfm','auto',str(data/'images'),'-o',str(output),'--quality','extreme','--camera-mode','single',
          '--camera-model',config.get('camera_model','opencv'),'--mapper','bottom-up','--pairs','prefilter',
          '--prefilter-neighbors','64','--max-image-size',str(config.get('sfm_max_image_size',1920)),
          '--masks',str(data/'masks'),'--threads',str(config.get('threads',8)),'--decode-threads','8','--device',device,
          '--sensor-gauge','none','--metric-gps','none','--exif-attitude','none']
    if config.get('ordered_images',False):args+=['--data-type','video','--sequence','.','--prefilter-sequential','--overlap','30']
    else:args+=['--data-type','individual']
    return args


def geometry_command(binary,data,maximum,device):
    return [binary,'geometry',str(data),'--model','moge2-vitl','--max-size',str(maximum),'--num-tokens','3600',
            '--depth','--normal-format','png','--depth-units','relative','--ray-depth','auto','--device',device]


def training_command(binary,data,root,plan,device,resume=False):
    args=[binary,'train','3dgs','--data',str(data),'--data-format','colmap','--colmap-recon-dir','sparse/0',
          '--device',device,'--quality','ultra','--output-dir-prefix',str(root/'training'),'--output-dir-name','final',
          '--num-iterations',str(plan['iterations']),'--cap-max',str(plan['gaussians']),'--sh-degree','0',
          '--sh-degree-warmup-every','0','--random-init','never','--train-resolution-divisor','1',
          '--load-masks','1','--apply-loss-for-mask','0','--load-depths','1','--load-normals','1',
          '--depth-supervision-weight','.01','--normal-supervision-weight','.01','--floater-suppression','mild',
          '--cache-images',plan['cache_images'],'--disable-viewer','1','--keep-viewer-alive','0',
          '--eval-mode','interval','--eval-interval','100','--save-eval-images','1','--steps-per-save','10000',
          '--save-only-latest-checkpoint','1','--save-full-checkpoint','1','--refine-stop-num-iter','10000',
          '--scene-center','none']
    if resume:args+=['--resume',str(root/'training/final')]
    return args


def command(root,name,args,env,lock_fd=None,gpu_index=None,resources=None):
    root=Path(root);(root/'logs').mkdir(exist_ok=True);write(root/'commands'/(name+'.json'),args)
    kwargs=dict(cwd=root,env=env,stdin=subprocess.DEVNULL)
    if os.name=='posix':kwargs.update(start_new_session=True,pass_fds=(() if lock_fd is None else tuple(lock_fd) if isinstance(lock_fd,(tuple,list)) else (lock_fd,)))
    elif os.name=='nt':kwargs['creationflags']=subprocess.CREATE_NO_WINDOW
    resources=resources or {};last_gpu_query=0
    with (root/'logs'/(name+'.log')).open('ab') as log:
        process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,**kwargs)
        try:
            while process.poll() is None:
                if (root/'STOP').exists():raise InterruptedError('STOP requested')
                if gpu_index is not None and time.monotonic()-last_gpu_query>15:
                    last_gpu_query=time.monotonic()
                    try:memory=gpu_info(gpu_index)
                    except (subprocess.SubprocessError,OSError):memory=None
                    if memory and memory['free_gib']<max(1.,memory['total_gib']*resources.get('free_reserve_fraction',.2)):
                        raise MemoryError('GPU free reserve exhausted; stopping only this job; checkpoint retained')
                current={}
                if os.name=='posix':
                    try:
                        for line in Path(f'/proc/{process.pid}/status').read_text().splitlines():
                            if line.startswith('VmRSS:'):current['child_rss_gib']=int(line.split()[1])/1024**2
                    except FileNotFoundError:pass
                    if current.get('child_rss_gib',0)>resources.get('ram_gib',64):raise MemoryError('Job RAM budget exceeded')
                status(root,name,child_pid=process.pid,log=str(root/'logs'/(name+'.log')),**current)
                time.sleep(2)
            if process.returncode:raise RuntimeError(f'{name} exited {process.returncode}; see logs/{name}.log')
        except BaseException:
            if process.poll() is None:
                if os.name=='posix':os.killpg(process.pid,signal.SIGTERM)
                else:process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name=='posix':os.killpg(process.pid,signal.SIGKILL)
                    else:process.kill()
                    process.wait()
            raise


def plan_budget(audit,image_records,memory,config):
    import math
    resources=config.get('resources',{});limits=config.get('training',{})
    pixels=max(r['width']*r['height'] for r in image_records);rgb=sum(r['width']*r['height']*3 for r in image_records)/2**30
    free=min(memory['free_gib']*.8,resources.get('gpu_gib',18),memory['free_gib']-memory['total_gib']*resources.get('free_reserve_fraction',.2))
    reserve=max(4.,pixels/1e6*1.2);maximum=int(max(0.,free-reserve)*2**30/1100)
    minimum=max(500000,audit['points'])
    target=max(minimum,audit['registered_images']*4000,audit['points']*5)
    gaussians=min(int(limits.get('max_gaussians',15000000)),maximum,int(limits.get('gaussians',target)))
    if gaussians<minimum:raise MemoryError('Insufficient GPU memory for initialization and rendering reserve')
    steps=math.ceil((30000+1400*math.sqrt(audit['registered_images']))/10000)*10000
    steps=int(limits.get('iterations',min(int(limits.get('max_iterations',120000)),steps)))
    if steps<10000:raise ValueError('At least 10,000 iterations required')
    ram=resources.get('ram_gib',64)
    if ram<12:raise ValueError('At least 12GiB RAM budget required')
    return dict(gaussians=gaussians,iterations=steps,cache_images='disk' if rgb*4+12>ram else 'cpu',
                sh_degree=0,depth_weight=.01,normal_weight=.01,floater_suppression='mild',
                estimated_bytes_per_gaussian=1100,reserve_gib=reserve,policy='bounded heuristic; not a quality or peak-memory guarantee')
