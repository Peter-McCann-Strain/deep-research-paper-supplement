"""JSON checkpoint save/load for pattern state."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from deep_research.config import CHECKPOINTS_DIR

log = structlog.get_logger()


class StateManager:
    """Save and load pattern state as JSON checkpoints."""

    def __init__(self, pattern_name: str, run_id: str = ""):
        self.pattern_name = pattern_name
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._dir = CHECKPOINTS_DIR / pattern_name / self.run_id
        self._dir.mkdir(parents=True, exist_ok=True)

    def save(self, stage: str, data: Dict[str, Any]) -> Path:
        """Save a checkpoint for a given stage."""
        path = self._dir / f"{stage}.json"
        payload = {
            "pattern": self.pattern_name,
            "run_id": self.run_id,
            "stage": stage,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data,
        }
        path.write_text(json.dumps(payload, default=str, indent=2))
        log.debug("checkpoint_saved", stage=stage, path=str(path))
        return path

    def load(self, stage: str) -> Optional[Dict[str, Any]]:
        """Load a checkpoint for a given stage."""
        path = self._dir / f"{stage}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text())
        log.debug("checkpoint_loaded", stage=stage)
        return payload.get("data")

    def has_checkpoint(self, stage: str) -> bool:
        return (self._dir / f"{stage}.json").exists()

    def list_stages(self) -> list[str]:
        """List all saved stages."""
        return [p.stem for p in sorted(self._dir.glob("*.json"))]

    @staticmethod
    def list_runs(pattern_name: str) -> list[str]:
        """List all runs for a pattern."""
        d = CHECKPOINTS_DIR / pattern_name
        if not d.exists():
            return []
        return sorted(p.name for p in d.iterdir() if p.is_dir())
