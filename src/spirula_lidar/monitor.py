"""Read-only HTTP polling. No background model calls or external assets."""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from .artifacts import read

PAGE='''<!doctype html><meta charset="utf-8"><title>Spirula + LiDAR</title><style>body{max-width:1100px;margin:30px auto;font:16px system-ui;background:#f5f7fa;color:#17212e}pre{white-space:pre-wrap;background:white;padding:20px;border-radius:10px}h1{font-size:24px}#connection{color:#526070}</style><h1>Spirula + LiDAR 任务状态</h1><p id="connection">正在连接</p><pre id="status"></pre><h2>最近日志</h2><pre id="log"></pre><script>
async function tick(){try{let r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error(r.status);let d=await r.json();document.querySelector('#connection').textContent='读取成功 · '+new Date().toLocaleString()+' · 状态文件更新时间 '+(d.status.updated_at?new Date(d.status.updated_at*1000).toLocaleString():'未知');document.querySelector('#status').textContent=JSON.stringify(d.status,null,2);document.querySelector('#log').textContent=d.log||'当前阶段没有子进程日志';}catch(e){document.querySelector('#connection').textContent='暂时无法读取：'+e.message+'，5秒后重试';}}tick();setInterval(tick,5000);</script>'''


def snapshot(root):
    root=Path(root).resolve();state=read(root/'status.json') if (root/'status.json').exists() else dict(status='unknown');log=''
    if state.get('log'):
        path=Path(state['log']).resolve()
        if root in path.parents and path.is_file():
            with path.open('rb') as stream:
                stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-12000));log=stream.read().decode('utf8','replace')
    return dict(status=state,log=log)


def serve(root,host='127.0.0.1',port=8765):
    root=Path(root).resolve()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path=='/':body=PAGE.encode();kind='text/html; charset=utf-8'
            elif self.path=='/api/status':body=json.dumps(snapshot(root),ensure_ascii=False).encode();kind='application/json; charset=utf-8'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def log_message(self,*args):pass
    print(f'Monitor: http://{host}:{port} (read-only)',flush=True)
    ThreadingHTTPServer((host,port),Handler).serve_forever()
