"""Constrained YAML, frozen run config, and path checks."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
PROJECT_ROOT = Path("/projects/b5ac/project_2")
RESULTS_ROOT = PROJECT_ROOT / "results" / "03_integrated_analysis"


def load_yaml(path: Path) -> dict:
    return _parse_yaml(path.read_text(encoding="utf-8"))


def _parse_scalar(raw: str):
    text = raw.strip()
    if text == "" or text in {"null", "~"}:
        return None
    if text in {"true", "false"}:
        return text == "true"
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    return text


def _parse_yaml(text: str):
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))
    value, _ = _block(lines, 0, 0)
    return value


def _block(lines, index, indent):
    if index >= len(lines):
        return {}, index
    if lines[index][1].startswith("- "):
        return _list(lines, index, indent)
    return _map(lines, index, indent)


def _map(lines, index, indent):
    out = {}
    while index < len(lines):
        level, text = lines[index]
        if level < indent or text.startswith("- "):
            break
        if level > indent:
            raise ValueError(f"unexpected indent: {text}")
        key, _, rest = text.partition(":")
        rest = rest.strip()
        index += 1
        if rest:
            out[key] = _parse_scalar(rest)
        elif index < len(lines) and lines[index][0] > level:
            child, index = _block(lines, index, lines[index][0])
            out[key] = child
        else:
            out[key] = None
    return out, index


def _list(lines, index, indent):
    out = []
    while index < len(lines):
        level, text = lines[index]
        if level < indent or not text.startswith("- "):
            break
        item = text[2:].strip()
        index += 1
        if item and ":" in item and not item.startswith("["):
            key, _, rest = item.partition(":")
            node = {key: _parse_scalar(rest) if rest else None}
            while index < len(lines) and lines[index][0] > level and not lines[index][1].startswith("- "):
                child_level, child = lines[index]
                ckey, _, crest = child.partition(":")
                index += 1
                if crest:
                    node[ckey] = _parse_scalar(crest)
                elif index < len(lines) and lines[index][0] > child_level:
                    nested, index = _block(lines, index, lines[index][0])
                    node[ckey] = nested
                else:
                    node[ckey] = None
            out.append(node)
        elif item:
            out.append(_parse_scalar(item))
        elif index < len(lines) and lines[index][0] > level:
            nested, index = _block(lines, index, lines[index][0])
            out.append(nested)
        else:
            out.append(None)
    return out, index


def load_bundle(config_dir: Path | None = None) -> dict:
    root = config_dir or (PACKAGE / "config")
    pilot = load_yaml(root / "pilot.yaml")
    resources = load_yaml(root / "resources.yaml")
    registry = load_yaml(root / "input_registry.yaml")
    return {"pilot": pilot, "resources": resources, "registry": registry}


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def redact(bundle: dict) -> dict:
    copied = json.loads(canonical(bundle))
    network = copied.get("resources", {}).get("network", {})
    for key in ("openai_key_env", "ncbi_key_env"):
        name = network.get(key)
        if name:
            network[f"{key}_set"] = bool(os.environ.get(name))
    return copied


def new_run_id(bundle: dict, now: datetime | None = None) -> str:
    moment = now or datetime.now(timezone.utc)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{digest(bundle)[:8]}"


def run_root(run_id: str) -> Path:
    return RESULTS_ROOT / "pilot_study_v1" / "runs" / run_id


def prompt_hashes() -> dict:
    out = {}
    for path in sorted((PACKAGE / "prompts").glob("*.txt")):
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((PACKAGE / "schemas").glob("*.json")):
        out[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def budget_state(bundle: dict) -> dict:
    budget = bundle["resources"]["budget"]
    approved = budget.get("approved") is True and budget.get("max_api_calls") not in (None, 0)
    return {
        "approved": approved,
        "max_api_calls": budget.get("max_api_calls"),
        "max_input_tokens": budget.get("max_input_tokens"),
        "max_output_tokens": budget.get("max_output_tokens"),
        "max_usd": budget.get("max_usd"),
        "reason": None if approved else "no_approved_call_and_token_budget",
    }
