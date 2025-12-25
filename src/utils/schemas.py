from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict

class PlanJSON(TypedDict, total=False):
    run_id: str
    prompt_id: str
    assumptions: List[str]
    modeling_steps: List[str]
    parameters: Dict[str, Any]
    constraints: Dict[str, Any]
    output_targets: Dict[str, str]  # e.g. {"cad_script": "...", "step": "..."}
    test_strategy: Dict[str, Any]   # for 4A

class VerifyReportJSON(TypedDict, total=False):
    status: str  # pass / fail
    issues: List[Dict[str, Any]]
    evidence: Dict[str, Any]

class OptPatchJSON(TypedDict, total=False):
    status: str  # "back_to_plan" | "back_to_script"
    rationale: str
    patch: Dict[str, Any]           # e.g. patch to plan or hints to script
    next_step: str                  # "1" or "2"
