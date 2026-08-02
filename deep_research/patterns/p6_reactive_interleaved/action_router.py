"""Parse the LLM's JSON action output and dispatch to the correct handler.

The agent model returns a JSON blob like::

    {"action": "SEARCH", "params": {"queries": ["q1", "q2"]}}

This module parses that (tolerating markdown code fences and minor
formatting quirks) and returns a normalised ``(action_type, params)``
tuple that the main loop can switch on.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import structlog

log = structlog.get_logger()

# Canonical action names
SEARCH = "SEARCH"
DEEP_READ = "DEEP_READ"
DRAFT = "DRAFT"
REFLECT = "REFLECT"
FINALIZE = "FINALIZE"

VALID_ACTIONS = {SEARCH, DEEP_READ, DRAFT, REFLECT, FINALIZE}


def parse_action(raw_output: str) -> Tuple[str, Dict[str, Any]]:
    """Parse the LLM's text output into ``(action_type, params)``.

    Falls back to ``REFLECT`` with empty params if the JSON is unparseable,
    so the loop always has a safe next step.
    """
    text = raw_output.strip()

    # Strip markdown code fences (```json ... ```)
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Attempt direct parse
    data = _try_parse_json(text)
    if data is None:
        log.warning("action_parse_failed_fallback_reflect", raw=text[:200])
        return REFLECT, {}

    action = str(data.get("action", "")).upper().strip()
    params = data.get("params", {})
    if not isinstance(params, dict):
        params = {}

    if action not in VALID_ACTIONS:
        log.warning("unknown_action_fallback_reflect", action=action)
        return REFLECT, {}

    log.debug("action_parsed", action=action, params_keys=list(params.keys()))
    return action, params


def _try_parse_json(text: str) -> Dict[str, Any] | None:
    """Attempt to parse JSON, trying progressively looser strategies."""
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return None
