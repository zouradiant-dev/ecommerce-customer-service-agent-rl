from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .models import TaskSpec
from .simulator import ReferenceAgent, WeakBaselineAgent, run_task
from .tasks import TEMPLATE_SPLITS, build_task_dataset, load_task_paths


class DemoApplication:
    """Dependency-free application service for inspecting verified rollouts."""

    def __init__(self, output_dir: Path, *, replicas: int = 1, seed: int = 7) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.dataset_dir = self.output_dir / "dataset"
        self.manifest = self.dataset_dir / "tasks.jsonl"
        summary_path = self.dataset_dir / "summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                summary = {}
        stale = (
            not self.manifest.exists()
            or int(summary.get("templates") or 0) != len(TEMPLATE_SPLITS)
            or int(summary.get("replicas_per_template") or 0) != replicas
            or int(summary.get("seed") or -1) != seed
        )
        if stale:
            build_task_dataset(self.dataset_dir, replicas=replicas, seed=seed)
        self._task_paths: dict[str, Path] = {}
        for path in load_task_paths(self.manifest):
            task = TaskSpec.from_file(path)
            self._task_paths[task.task_id] = path

    def list_tasks(self) -> list[dict[str, Any]]:
        split_order = {"train": 0, "dev": 1, "test": 2}
        tasks = [TaskSpec.from_file(path) for path in self._task_paths.values()]
        tasks.sort(key=lambda task: (split_order.get(task.split, 9), task.scenario, task.task_id))
        return [
            {
                "task_id": task.task_id,
                "scenario": task.scenario,
                "split": task.split,
                "user_prompt": task.user_prompt,
                "required_tools": list(task.required_tools),
                "requires_followup": bool(task.metadata.get("scripted_user_replies")),
            }
            for task in tasks
        ]

    def run(self, task_id: str, agent_name: str) -> dict[str, Any]:
        if task_id not in self._task_paths:
            raise KeyError("unknown_task_id")
        agents = {
            "reference": ReferenceAgent,
            "baseline": WeakBaselineAgent,
        }
        if agent_name not in agents:
            raise ValueError("unknown_agent")
        task = TaskSpec.from_file(self._task_paths[task_id])
        return run_task(task, agents[agent_name]()).to_dict()


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], application: DemoApplication) -> None:
        self.application = application
        super().__init__(address, DemoRequestHandler)


class DemoRequestHandler(BaseHTTPRequestHandler):
    server: DemoServer

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/":
            self._send(HTTPStatus.OK, DEMO_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if route == "/api/tasks":
            self._send_json(HTTPStatus.OK, {"tasks": self.server.application.list_tasks()})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/run":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 65_536:
                raise ValueError("invalid_content_length")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = self.server.application.run(
                str(payload.get("task_id") or ""),
                str(payload.get("agent") or "reference"),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, result)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"web-demo {self.address_string()} {format % args}")

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status.value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)


DEMO_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Agent RL 审计台</title>
  <style>
    :root { color-scheme: dark; --bg:#09111f; --panel:#101c30; --line:#243650; --ink:#eaf2ff;
      --muted:#90a3bd; --cyan:#4fd1c5; --gold:#f7c96b; --red:#fb7185; --green:#86efac; }
    * { box-sizing:border-box } body { margin:0; font:14px/1.55 Inter,system-ui,sans-serif;
      background:radial-gradient(circle at 15% 0,#173558 0,transparent 35%),var(--bg); color:var(--ink) }
    main { width:min(1180px,94vw); margin:36px auto 64px } header { display:flex; justify-content:space-between;
      gap:24px; align-items:end; margin-bottom:22px } h1 { margin:0; font-size:clamp(26px,4vw,44px); letter-spacing:-.04em }
    .eyebrow { color:var(--cyan); font-weight:750; letter-spacing:.12em; text-transform:uppercase }
    .muted { color:var(--muted) } .panel { background:color-mix(in srgb,var(--panel) 92%,transparent);
      border:1px solid var(--line); border-radius:18px; padding:20px; box-shadow:0 18px 50px #0005 }
    .controls { display:grid; grid-template-columns:1fr 180px auto; gap:12px; margin-bottom:18px }
    select,button { min-height:44px; border-radius:10px; border:1px solid var(--line); color:var(--ink);
      background:#0b1729; padding:0 13px; font:inherit } button { cursor:pointer; font-weight:750;
      background:linear-gradient(135deg,#22a699,#2674c9); border:0; min-width:130px }
    button:disabled { opacity:.55; cursor:wait } .prompt { padding:14px; border-left:3px solid var(--gold);
      background:#0a1526; border-radius:8px; min-height:56px }
    .metrics { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:18px 0 }
    .metric { background:#0a1526; border:1px solid var(--line); border-radius:12px; padding:13px }
    .metric span { display:block; color:var(--muted); font-size:12px }.metric strong { font-size:21px }
    .pass { color:var(--green) }.fail { color:var(--red) }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px }.stack { display:grid; gap:10px }
    h2 { margin:0 0 12px; font-size:16px } .tool { padding:11px; background:#0a1526; border-radius:10px;
      border:1px solid var(--line) }.tool-head { display:flex; justify-content:space-between; gap:12px }
    code,pre { font:12px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace } pre { white-space:pre-wrap;
      word-break:break-word; color:#bdd0e7; margin:8px 0 0 }.final { font-size:16px; color:#fff }
    .tag { display:inline-block; padding:2px 7px; border-radius:999px; background:#20324b; color:#b9cde6; font-size:11px }
    .empty { padding:38px; text-align:center; color:var(--muted) }
    @media(max-width:800px){.controls,.grid{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}header{align-items:start;flex-direction:column}}
  </style>
</head>
<body><main>
  <header><div><div class="eyebrow">Verifiable agent environment</div><h1>电商售后 Agent RL 审计台</h1></div>
    <div class="muted">SQLite 最终状态 × 工具轨迹 × 政策约束</div></header>
  <section class="panel">
    <div class="controls"><select id="task"></select><select id="agent"><option value="reference">Reference（正确策略）</option>
      <option value="baseline">Weak baseline（故障对照）</option></select><button id="run">运行隔离任务</button></div>
    <div id="prompt" class="prompt muted">正在加载任务…</div>
  </section>
  <div id="result" class="empty">选择场景后运行，页面会展示奖励分解、验证结果与完整工具审计轨迹。</div>
</main>
<script>
const $=id=>document.getElementById(id); let tasks=[];
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function current(){return tasks.find(x=>x.task_id===$('task').value)}
function prompt(){const t=current(); $('prompt').innerHTML=t?`<span class="tag">${esc(t.split)}</span> <span class="tag">${esc(t.scenario)}</span><br>${esc(t.user_prompt)}`:'无任务'}
async function load(){const r=await fetch('/api/tasks'); tasks=(await r.json()).tasks; $('task').innerHTML=tasks.map(t=>`<option value="${esc(t.task_id)}">[${esc(t.split)}] ${esc(t.scenario)}</option>`).join(''); prompt()}
function render(d){const c=d.reward_components||{},v=d.verifier||{},tools=d.tool_executions||[];
 const metric=(name,val,ok)=>`<div class="metric"><span>${name}</span><strong class="${ok?'pass':'fail'}">${esc(val)}</strong></div>`;
 const trail=tools.map((t,i)=>`<div class="tool"><div class="tool-head"><strong>${i+1}. ${esc(t.name)}</strong><span class="${t.ok?'pass':'fail'}">${t.ok?'成功':'失败'}</span></div><pre>参数 ${esc(JSON.stringify(t.arguments,null,2))}\n结果 ${esc(JSON.stringify(t.output,null,2))}</pre></div>`).join('')||'<div class="muted">没有工具调用</div>';
 $('result').className=''; $('result').innerHTML=`<div class="metrics">${metric('严格奖励',d.reward,d.reward===1)}${metric('状态',c.state,c.state===1)}${metric('沟通',c.communication,c.communication===1)}${metric('政策安全',c.policy,c.policy===1)}${metric('必要工具',c.required_tools,c.required_tools===1)}</div><div class="grid"><section class="panel"><h2>最终回复</h2><div class="final">${esc(d.final_response)||'—'}</div><h2 style="margin-top:20px">Verifier</h2><pre>${esc(JSON.stringify(v,null,2))}</pre></section><section class="panel"><h2>工具审计轨迹 · ${tools.length} 步</h2><div class="stack">${trail}</div></section></div>`}
$('task').addEventListener('change',prompt); $('run').addEventListener('click',async()=>{const b=$('run');b.disabled=true;b.textContent='运行中…';try{const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id:$('task').value,agent:$('agent').value})});const d=await r.json();if(!r.ok)throw Error(d.error||'运行失败');render(d)}catch(e){$('result').className='empty fail';$('result').textContent=e.message}finally{b.disabled=false;b.textContent='运行隔离任务'}}); load().catch(e=>{$('prompt').textContent=e.message});
</script></body></html>"""


def serve_demo(
    output_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    replicas: int = 1,
    seed: int = 7,
) -> None:
    application = DemoApplication(output_dir, replicas=replicas, seed=seed)
    server = DemoServer((host, port), application)
    print(f"Demo ready at http://{host}:{port} ({len(application.list_tasks())} tasks)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
