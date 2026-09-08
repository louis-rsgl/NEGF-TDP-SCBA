"""Append-only campaign log and atomically refreshed HTML dashboard."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping

from psscba.frontend.storage import atomic_json, atomic_text


SOFTWARE_NAME = "NEGF TDP PS-SCBA"
BANNER = r"""
 _   _ _____ ____ _____   _____ ____  ____    ____  ____        ____   ____ ____    _
| \ | | ____/ ___|  ___| |_   _|  _ \|  _ \  |  _ \/ ___|      / ___| / ___| __ )  / \
|  \| |  _|| |  _| |_      | | | | | | |_) | | |_) \___ \ _____\___ \| |   |  _ \ / _ \
| |\  | |__| |_| |  _|     | | | |_| |  __/  |  __/ ___) |_____|___) | |___| |_) / ___ \
|_| \_|_____\____|_|       |_| |____/|_|     |_|   |____/      |____/ \____|____/_/   \_\
""".strip("\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "--:--:--"
    value = max(0, int(round(float(seconds))))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _loss(value: float | None) -> str:
    return "pending" if value is None else f"{float(value):.3e}"


def read_job_snapshot(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    status: dict[str, Any] = {}
    progress: dict[str, Any] = {}
    for path, destination in (
        (run_dir / "status.json", status),
        (run_dir / "progress.json", progress),
    ):
        if path.exists():
            try:
                destination.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
    return {
        "status": status.get("status", progress.get("state", "created")),
        "stage": progress.get("phase_label", progress.get("phase", status.get("phase", "waiting"))),
        "outer_step": progress.get("outer_iteration_display", status.get("iteration")),
        "outer_max": progress.get("outer_max_iterations"),
        "loss": progress.get("last_completed_loss"),
        "elapsed_seconds": progress.get("elapsed_seconds"),
        "rss_bytes": progress.get("rss_bytes"),
        "residual_retarded": progress.get("residual_retarded"),
        "residual_lesser": progress.get("residual_lesser"),
        "pending_residual_retarded": progress.get("pending_residual_retarded"),
        "pending_residual_lesser": progress.get("pending_residual_lesser"),
        "projected_green_retarded_change": progress.get("projected_green_retarded_change"),
        "projected_green_lesser_change": progress.get("projected_green_lesser_change"),
        "projected_occupation": progress.get("projected_occupation"),
        "sigma_hartree_installed": progress.get("sigma_hartree_installed"),
        "observable_time_nat": progress.get("observable_time_nat"),
        "observable_time_ps": progress.get("observable_time_ps"),
        "current_L_uA": progress.get("current_L_uA"),
        "current_R_uA": progress.get("current_R_uA"),
        "batch_completed": progress.get("batch_completed"),
        "batch_total": progress.get("batch_total"),
        "batch_fraction": progress.get("batch_fraction"),
        "scba_mode": progress.get("scba_mode"),
        "timestamp": progress.get("timestamp", utc_now()),
    }


class CampaignDashboard:
    """Thread-safe campaign summary that never receives detailed solver lines."""

    def __init__(self, campaign_dir: str | Path, manifest: Mapping[str, Any]) -> None:
        self.campaign_dir = Path(campaign_dir)
        self.manifest = dict(manifest)
        self._lock = threading.RLock()
        self.campaign_dir.mkdir(parents=True, exist_ok=True)

    @property
    def jobs(self) -> list[dict[str, Any]]:
        return list(self.manifest.get("jobs", []))

    def start(self) -> None:
        kind = str(self.manifest.get("kind", "sweep")).replace("_", " ")
        lines = [BANNER, "", f"{SOFTWARE_NAME} {kind}", f"started={utc_now()}"]
        lines.append(f"jobs={len(self.jobs)}  workers={self.manifest.get('max_workers', 1)}")
        lines.append("Detailed SCBA stages and loss histories live under jobs/<case-id>/solver.log.")
        self._append("\n".join(lines) + "\n")
        self.refresh(reason="campaign initialized")

    def log_message(self, message: str) -> None:
        """Append one campaign-level message without solver-stage detail."""
        self._append(message.rstrip("\n") + "\n")

    def _append(self, payload: str) -> None:
        path = self.campaign_dir / "main.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()

    def update_status(self, case_id: str, status: str, **details: Any) -> None:
        with self._lock:
            for job in self.jobs:
                if job["case_id"] == case_id:
                    job["status"] = status
                    job.update(details)
                    job["updated_at"] = utc_now()
                    break
            self.manifest["jobs"] = self.jobs
            self.manifest["updated_at"] = utc_now()
            atomic_json(self.campaign_dir / "campaign.json", self.manifest)

    def refresh(self, *, reason: str = "heartbeat") -> None:
        with self._lock:
            rows: list[dict[str, Any]] = []
            for job in self.jobs:
                snapshot = read_job_snapshot(job["run_directory"])
                if job.get("status") in {
                    "accepted",
                    "complete",
                    "failed",
                    "interrupted",
                }:
                    snapshot["status"] = job["status"]
                merged = {**job, **snapshot}
                rows.append(merged)
                job["status"] = merged["status"]
                job["updated_at"] = merged["timestamp"]
            self.manifest["jobs"] = self.jobs
            self.manifest["updated_at"] = utc_now()
            atomic_json(self.campaign_dir / "campaign.json", self.manifest)
            self._append(self._text_snapshot(rows, reason))
            atomic_text(self.campaign_dir / "dashboard.html", self._html(rows))

    def finish(self) -> None:
        kind = str(self.manifest.get("kind", "sweep")).replace("_", " ")
        self.refresh(reason=f"{kind} finished")
        states: dict[str, int] = {}
        for job in self.jobs:
            states[job.get("status", "unknown")] = states.get(job.get("status", "unknown"), 0) + 1
        atomic_json(self.campaign_dir / "summary.json", {
            "software": SOFTWARE_NAME,
            "finished_at": utc_now(),
            "states": states,
            "total_jobs": len(self.jobs),
        })

    @staticmethod
    def _text_snapshot(rows: Iterable[Mapping[str, Any]], reason: str) -> str:
        lines = ["", f"[{utc_now()}] {reason}"]
        lines.append("CASE                         STATE       OUTER        PHASE/PROGRESS                 RΣR       RΣ<       JL(µA)    JR(µA)   t(ps)    WALL      RSS")
        lines.append("-" * 156)
        for row in rows:
            step = row.get("outer_step")
            maximum = row.get("outer_max")
            scf = "--" if step is None else f"{int(step):04d}/{int(maximum or 0):04d}"
            rss = row.get("rss_bytes")
            rss_text = "--" if rss is None else f"{float(rss) / 1024**3:.1f}G"
            rr = "exact g=0" if row.get("scba_mode") == "noninteracting_exact" else _loss(row.get("residual_retarded"))
            rl = "exact g=0" if row.get("scba_mode") == "noninteracting_exact" else _loss(row.get("residual_lesser"))
            jl = _loss(row.get("current_L_uA"))
            jr = _loss(row.get("current_R_uA"))
            tps = "--" if row.get("observable_time_ps") is None else f"{float(row['observable_time_ps']):.3f}"
            progress_text = str(row.get("stage", "waiting"))[:26]
            if row.get("batch_total"):
                progress_text += f" {int(row.get('batch_completed') or 0)}/{int(row['batch_total'])}"
            lines.append(
                f"{str(row['case_id'])[:28]:28} {str(row.get('status', 'unknown'))[:11]:11} "
                f"{scf:11} {progress_text:27} {rr:9} {rl:9} {jl:9} {jr:9} {tps:>7} "
                f"{_duration(row.get('elapsed_seconds')):9} {rss_text:>5}"
            )
        return "\n".join(lines) + "\n"

    def _html(self, rows: Iterable[Mapping[str, Any]]) -> str:
        body = []
        for row in rows:
            status = str(row.get("status", "unknown"))
            batch_completed = row.get("batch_completed")
            batch_total = row.get("batch_total")
            batch_text = (
                f"{batch_completed}/{batch_total}"
                if batch_completed is not None and batch_total is not None
                else "--/--"
            )
            body.append(
                "<tr>"
                f"<td>{escape(str(row['case_id']))}</td>"
                f"<td class='state {escape(status)}'>{escape(status)}</td>"
                f"<td>{escape(str(row.get('tier', '')))}</td>"
                f"<td>{escape(str(row.get('family', '')))}</td>"
                f"<td>{escape(str(row.get('protocol', '')))}</td>"
                f"<td>{row.get('bandwidth', '')}</td><td>{row.get('phonon_energy', '')}</td>"
                f"<td>{row.get('coupling_meV', '')}</td>"
                f"<td>{escape(str(row.get('outer_step') if row.get('outer_step') is not None else '--'))}</td>"
                f"<td>{escape(str(row.get('stage', 'waiting')))}"
                f" ({batch_text})</td>"
                f"<td>{escape(_loss(row.get('residual_retarded')))}</td>"
                f"<td>{escape(_loss(row.get('residual_lesser')))}</td>"
                f"<td>{escape(_loss(row.get('current_L_uA')))}</td>"
                f"<td>{escape(_loss(row.get('current_R_uA')))}</td>"
                f"<td>{'--' if row.get('observable_time_ps') is None else f'{float(row.get('observable_time_ps')):.3f}'}</td>"
                f"<td>{_duration(row.get('elapsed_seconds'))}</td>"
                "</tr>"
            )
        preflight = self.manifest.get("preflight")
        if isinstance(preflight, Mapping):
            preflight_text = (
                f"preflight: {preflight.get('protocol', '--')} / "
                f"{preflight.get('stage', '--')}"
            )
        else:
            preflight_text = "preflight: idle"
        return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="10">
<title>{SOFTWARE_NAME}</title><style>
:root{{--bg:#002b36;--panel:#073642;--text:#eee8d5;--muted:#93a1a1;--accent:#d77b5d;--ok:#2aa198;--bad:#dc322f;}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px ui-monospace,SFMono-Regular,Consolas,monospace;}}
main{{max-width:1500px;margin:auto;padding:28px;}} pre{{color:var(--accent);font-weight:700;line-height:1.05;}}
h1{{color:var(--accent);letter-spacing:.08em}} .meta{{color:var(--muted);margin-bottom:18px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);box-shadow:0 10px 30px #001f27;}}
th,td{{padding:9px 11px;border-bottom:1px solid #28505a;text-align:left;white-space:nowrap;}}
th{{color:var(--accent)}} .state.complete,.state.accepted{{color:var(--ok)}} .state.failed{{color:var(--bad)}}
</style></head><body><main><pre>{escape(BANNER)}</pre><h1>{SOFTWARE_NAME}</h1>
<div class="meta">updated {escape(str(self.manifest.get('updated_at', utc_now())))} · {escape(preflight_text)} · detailed logs are stored per job</div>
<table><thead><tr><th>Case</th><th>State</th><th>Tier</th><th>Family</th><th>Protocol</th><th>W/Γ</th><th>ω₀/Γ</th><th>g meV</th><th>Outer</th><th>Phase/progress</th><th>RΣR</th><th>RΣ&lt;</th><th>J_L (µA)</th><th>J_R (µA)</th><th>t (ps)</th><th>Wall</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></main></body></html>"""
