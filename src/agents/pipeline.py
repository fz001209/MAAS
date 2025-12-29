from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.fs import ensure_dir, write_json, write_text, read_json, copy_file, copy_tree
from src.utils.events import build_event
from src.utils.render_stub import list_rendered_images

def _extract_first_json(text: str) -> Dict[str, Any]:
    """
    允许模型输出纯 JSON，或夹带少量文字；这里做稳健提取。
    """
    text = text.strip()
    # already json
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    # find first {...}
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output.")
    return json.loads(m.group(0))

def _safe_run(cmd: List[str], cwd: str | Path, timeout: int = 120) -> Tuple[int, str]:
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        text=True,
        check=False,
    )
    return p.returncode, p.stdout

def stage_paths(base_run_dir: Path) -> Dict[str, Path]:
    return {
        "run": base_run_dir,
        "input": ensure_dir(base_run_dir / "input"),
        "artifacts": ensure_dir(base_run_dir / "artifacts"),
        "memory": ensure_dir(base_run_dir / "memory"),
        "events": ensure_dir(base_run_dir / "memory" / "events"),
    }

def agent1_planner(run_id: str, paths: Dict[str, Path], anforderungsliste_path: Path, agent) -> Path:
    plan_path = paths["artifacts"] / "plan.json"
    event_path = paths["events"] / "event1.json"

    ev_start = build_event(
        run_id, "1", "Agent1_Planner", "start",
        inputs={"anforderungsliste": str(anforderungsliste_path)},
        outputs={"plan_json": str(plan_path), "event": str(event_path)},
    )
    write_json(event_path, ev_start)

    prompt = {
        "run_id": run_id,
        "input_file": str(anforderungsliste_path),
        "required_output_file": str(plan_path),
        "instructions": "Read the YAML file content and produce plan.json in strict JSON.",
    }
    resp = agent.generate_reply(messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}])
    plan = _extract_first_json(resp if isinstance(resp, str) else resp.get("content", ""))

    # enforce run_id and output targets
    plan["run_id"] = run_id
    plan.setdefault("output_targets", {})
    plan["output_targets"].update({
        "plan_json": str(plan_path),
        "cad_script": str(paths["artifacts"] / "cad_script.py"),
        "output_manifest": str(paths["artifacts"] / "output_manifest.json"),
        "verify_report": str(paths["artifacts"] / "verify_report.json"),
        "opt_patch": str(paths["artifacts"] / "opt_patch.json"),
    })

    write_json(plan_path, plan)

    ev_done = build_event(
        run_id, "1", "Agent1_Planner", "success",
        inputs={"anforderungsliste": str(anforderungsliste_path)},
        outputs={"plan_json": str(plan_path)},
        message="plan.json created",
    )
    write_json(event_path, ev_done)
    return plan_path

def agent2_cad_writer(run_id: str, paths: Dict[str, Path], plan_path: Path, agent) -> Path:
    cad_script_path = paths["artifacts"] / "cad_script.py"
    event_path = paths["events"] / "event2.json"

    write_json(event_path, build_event(
        run_id, "2", "Agent2_CADWriter", "start",
        inputs={"plan_json": str(plan_path)},
        outputs={"cad_script": str(cad_script_path), "event": str(event_path)},
    ))

    plan = read_json(plan_path)
    prompt = {
        "run_id": run_id,
        "plan_json_path": str(plan_path),
        "plan_json": plan,
        "required_output_file": str(cad_script_path),
        "instructions": (
            "Generate a single, complete CadQuery Python script.\n"
            "Hard requirements:\n"
            "1) Use `import cadquery as cq`\n"
            "2) The final model MUST be stored in a variable named `result` (a Workplane).\n"
            "3) Export STEP using `result.val().exportStep(step_path)` (NOT result.exportStep)\n"
            "4) Export STL using `result.val().exportStl(stl_path)`\n"
            "5) Write outputs into the same folder as this script (use Path(__file__).parent)\n"
            "6) If you add rendering, write PNGs into `render/` under output_dir.\n"
            "Return only Python code."
        ),
    }

    resp = agent.generate_reply(messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}])
    text = resp if isinstance(resp, str) else resp.get("content", "")

    # extract code block if present
    code = text
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.S)
    if m:
        code = m.group(1).strip()

    ensure_dir(paths["artifacts"] / "render")

    # -----------------------------
    # SAFETY PATCH: fix common CadQuery export mistakes
    # -----------------------------
    # 1) If LLM wrote result.exportStep(...), rewrite to result.val().exportStep(...)
    code = re.sub(r"\bresult\.exportStep\(", "result.val().exportStep(", code)
    code = re.sub(r"\bresult\.exportStl\(", "result.val().exportStl(", code)

    # 2) Ensure script uses output_dir = Path(__file__).resolve().parent
    #    If not present, prepend a minimal header.
    if "output_dir" not in code:
        header = (
            "from __future__ import annotations\n"
            "from pathlib import Path\n"
            "import cadquery as cq\n"
            "import os\n\n"
            "output_dir = Path(__file__).resolve().parent\n"
            "render_dir = output_dir / 'render'\n"
            "render_dir.mkdir(parents=True, exist_ok=True)\n\n"
        )
        # avoid double import cadquery if already exists
        if "import cadquery as cq" in code:
            # keep user's import, but still ensure output_dir exists
            header = (
                "from __future__ import annotations\n"
                "from pathlib import Path\n"
                "import os\n\n"
                "output_dir = Path(__file__).resolve().parent\n"
                "render_dir = output_dir / 'render'\n"
                "render_dir.mkdir(parents=True, exist_ok=True)\n\n"
            )
        code = header + code

    # 3) Ensure STEP/STL export exists; if not, append a footer that exports result
    must_have_step = re.search(r"\.exportStep\(", code) is not None
    must_have_stl = re.search(r"\.exportStl\(", code) is not None
    footer_lines = []
    if not must_have_step:
        footer_lines.append("step_path = str(output_dir / 'model.step')")
        footer_lines.append("result.val().exportStep(step_path)")
    if not must_have_stl:
        footer_lines.append("stl_path = str(output_dir / 'model.stl')")
        footer_lines.append("result.val().exportStl(stl_path)")
    if footer_lines:
        code = code.rstrip() + "\n\n# --- auto-added exports (pipeline safety) ---\n" + "\n".join(footer_lines) + "\n"

    write_text(cad_script_path, code)

    write_json(event_path, build_event(
        run_id, "2", "Agent2_CADWriter", "success",
        inputs={"plan_json": str(plan_path)},
        outputs={"cad_script": str(cad_script_path)},
        message="cad_script.py created",
    ))
    return cad_script_path

def agent3_executor(run_id: str, paths: Dict[str, Path], cad_script_path: Path, agent) -> Path:
    manifest_path = paths["artifacts"] / "output_manifest.json"
    exec_log_path = paths["artifacts"] / "exec.log.txt"
    event_path = paths["events"] / "event3.json"

    write_json(event_path, build_event(
        run_id, "3", "Agent3_Executor", "start",
        inputs={"cad_script": str(cad_script_path)},
        outputs={"output_manifest": str(manifest_path), "exec_log": str(exec_log_path), "event": str(event_path)},
    ))

    # IMPORTANT: always run with current venv python, and absolute script path
    script_abs = Path(cad_script_path).resolve()
    py_exe = sys.executable  # uses the current interpreter (venv)

    # execute script in artifacts dir (outputs land next to script)
    rc, out = _safe_run([py_exe, str(script_abs)], cwd=paths["artifacts"], timeout=180)
    write_text(exec_log_path, out)

    # collect artifacts
    render_dir = paths["artifacts"] / "render"
    imgs = list_rendered_images(render_dir)

    step_files = [str(p) for p in paths["artifacts"].glob("*.step")] + [str(p) for p in paths["artifacts"].glob("*.stp")]
    stl_files = [str(p) for p in paths["artifacts"].glob("*.stl")]

    manifest = {
        "run_id": run_id,
        "status": "success" if rc == 0 else "fail",
        "cad_script": str(script_abs),
        "step_files": step_files,
        "stl_files": stl_files,
        "render_images": imgs,
        "exec_log_path": str(exec_log_path),
        "artifacts_dir": str(paths["artifacts"]),
        "return_code": rc,
    }
    write_json(manifest_path, manifest)

    write_json(event_path, build_event(
        run_id, "3", "Agent3_Executor", "success" if rc == 0 else "fail",
        inputs={"cad_script": str(script_abs)},
        outputs={"output_manifest": str(manifest_path), "exec_log": str(exec_log_path)},
        message="execution finished",
        error="" if rc == 0 else f"Execution failed, return code {rc}",
    ))

    return manifest_path


def agent4a_verifier(run_id: str, paths: Dict[str, Path], plan_path: Path, manifest_path: Path, agent) -> Path:
    verify_path = paths["artifacts"] / "verify_report.json"
    event_path = paths["events"] / "event4A.json"

    write_json(event_path, build_event(
        run_id, "4A", "Agent4A_Verifier", "start",
        inputs={"plan_json": str(plan_path), "output_manifest": str(manifest_path)},
        outputs={"verify_report": str(verify_path), "event": str(event_path)},
    ))

    plan = read_json(plan_path)
    manifest = read_json(manifest_path)

    # Prepare multimodal messages if images exist
    image_paths = manifest.get("render_images", [])
    # If manifest has no images, also check artifacts/render
    if not image_paths:
        image_paths = list_rendered_images(paths["artifacts"] / "render")

    content_payload: List[Dict[str, Any]] = [{
        "type": "text",
        "text": json.dumps({
            "run_id": run_id,
            "plan_json": plan,
            "output_manifest": manifest,
            "instructions": (
                "Decide pass/fail. If images exist, use them to assess geometry + basic constraints/logic. "
                "If no images, do static validation on manifest vs plan (required files, expected outputs, etc.). "
                "Return STRICT JSON for verify_report.json."
            )
        }, ensure_ascii=False)
    }]

    # Attach images (for multimodal agents)
    for p in image_paths[:8]:  # limit
        content_payload.append({"type": "image_url", "image_url": {"url": f"file://{p}"}})

    resp = agent.generate_reply(messages=[{"role": "user", "content": content_payload}])
    text = resp if isinstance(resp, str) else resp.get("content", "")

    report = _extract_first_json(text)
    # enforce schema fields minimally
    report.setdefault("status", "fail")
    report.setdefault("issues", [])
    report.setdefault("evidence", {})
    report["evidence"].setdefault("render_images_used", image_paths)

    write_json(verify_path, report)

    write_json(event_path, build_event(
        run_id, "4A", "Agent4A_Verifier", "success",
        inputs={"plan_json": str(plan_path), "output_manifest": str(manifest_path)},
        outputs={"verify_report": str(verify_path)},
        message=f"verify status: {report.get('status')}",
    ))
    return verify_path

def agent5_optimizer(run_id: str, paths: Dict[str, Path], plan_path: Path, cad_script_path: Path, verify_path: Path, agent) -> Path:
    opt_path = paths["artifacts"] / "opt_patch.json"
    event_path = paths["events"] / "event5.json"

    write_json(event_path, build_event(
        run_id, "5", "Agent5_Optimizer", "start",
        inputs={"plan_json": str(plan_path), "cad_script": str(cad_script_path), "verify_report": str(verify_path)},
        outputs={"opt_patch": str(opt_path), "event": str(event_path)},
    ))

    plan = read_json(plan_path)
    verify = read_json(verify_path)

    prompt = {
        "run_id": run_id,
        "plan_json": plan,
        "verify_report": verify,
        "cad_script_path": str(cad_script_path),
        "instructions": (
            "If verify status is fail, produce opt_patch.json that decides next_step=1 or 2. "
            "If status is pass, still produce opt_patch.json with status='noop' and next_step='6'. "
            "Return STRICT JSON only."
        ),
    }
    resp = agent.generate_reply(messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}])
    patch = _extract_first_json(resp if isinstance(resp, str) else resp.get("content", ""))

    patch.setdefault("status", "noop")
    patch.setdefault("next_step", "6" if verify.get("status") == "pass" else "2")

    write_json(opt_path, patch)

    write_json(event_path, build_event(
        run_id, "5", "Agent5_Optimizer", "success",
        inputs={"verify_report": str(verify_path)},
        outputs={"opt_patch": str(opt_path)},
        message=f"optimizer next_step: {patch.get('next_step')}",
    ))

    return opt_path

def agent6_memory(
    run_id: str,
    paths: Dict[str, Path],
    user_input_path: Path,
    final_files_to_package: List[Path],
    agent,
) -> Tuple[Path, Path]:
    merged_events_path = paths["artifacts"] / "events_merged.json"
    final_zip_path = paths["artifacts"] / "final_model.zip"

    # ensure memory subfolders
    ensure_dir(paths["memory"] / "input")
    ensure_dir(paths["memory"] / "artifacts")

    # 1) copy all artifacts + input into memory
    copy_file(user_input_path, paths["memory"] / "input" / user_input_path.name)
    copy_tree(paths["artifacts"], paths["memory"] / "artifacts")

    # 2) merge events
    events_dir = paths["events"]
    event_files = sorted(events_dir.glob("event*.json"))
    merged = {
        "run_id": run_id,
        "events": [read_json(p) for p in event_files],
    }
    write_json(merged_events_path, merged)
    copy_file(merged_events_path, paths["memory"] / "events_merged.json")

    # 3) zip final package
    import zipfile
    with zipfile.ZipFile(final_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in final_files_to_package:
            if f.exists():
                z.write(f, arcname=f"artifacts/{f.name}")

        render_dir = paths["artifacts"] / "render"
        if render_dir.exists():
            for p in sorted(render_dir.glob("*.png")):
                z.write(p, arcname=f"artifacts/render/{p.name}")

        z.write(merged_events_path, arcname="events_merged.json")

    copy_file(final_zip_path, paths["memory"] / "final_model.zip")

    return final_zip_path, merged_events_path

