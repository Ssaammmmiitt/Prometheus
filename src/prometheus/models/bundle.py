"""A frozen, versioned model bundle: boosters, calibrators, and risk thresholds."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prometheus.config import load_settings
from prometheus.models.calibrate import Calibrator

MANIFEST = "manifest.json"


def bundles_root() -> Path:
    path = load_settings().paths.resolve("models") / "bundles"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class HorizonArtifacts:
    horizon: int
    model_file: str
    features: list[str]
    calibrator: Calibrator
    risk_thresholds: list[float]
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "model_file": self.model_file,
            "features": self.features,
            "calibrator": self.calibrator.to_dict(),
            "risk_thresholds": self.risk_thresholds,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> HorizonArtifacts:
        return cls(
            horizon=int(payload["horizon"]),
            model_file=payload["model_file"],
            features=list(payload["features"]),
            calibrator=Calibrator.from_dict(payload["calibrator"]),
            risk_thresholds=[float(v) for v in payload["risk_thresholds"]],
            metrics=payload.get("metrics", {}),
        )


@dataclass
class ModelBundle:
    """
    Everything needed to reproduce a prediction, pinned together.

    Boosters, calibrators, class thresholds, and the exact year split live in one
    versioned directory. Loading a bundle by version is the only supported way to
    predict, so a score can always be traced back to the artefacts that made it.
    """

    version: str
    horizons: dict[int, HorizonArtifacts]
    train_years: list[int]
    calibration_year: int
    test_year: int
    risk_class_names: list[str]
    risk_quantiles: list[float]
    created_at: str = ""
    notes: dict[str, Any] = field(default_factory=dict)
    root: Path | None = None

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at,
            "train_years": self.train_years,
            "calibration_year": self.calibration_year,
            "test_year": self.test_year,
            "risk_class_names": self.risk_class_names,
            "risk_quantiles": self.risk_quantiles,
            "notes": self.notes,
            "horizons": {str(h): a.to_dict() for h, a in sorted(self.horizons.items())},
        }

    def save(self, root: Path | None = None) -> Path:
        root = root or (bundles_root() / self.version)
        root.mkdir(parents=True, exist_ok=True)
        self.created_at = self.created_at or datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        for artifacts in self.horizons.values():
            source = Path(artifacts.model_file)
            target = root / f"lgbm_h{artifacts.horizon}.txt"
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
            artifacts.model_file = target.name
        (root / MANIFEST).write_text(
            json.dumps(self.to_dict(), indent=2), encoding="utf-8"
        )
        self.root = root
        return root

    @classmethod
    def load(cls, version: str | Path = "latest") -> ModelBundle:
        root = Path(version) if Path(str(version)).is_dir() else bundles_root() / str(version)
        if str(version) == "latest":
            candidates = sorted(
                p for p in bundles_root().iterdir() if (p / MANIFEST).is_file()
            )
            if not candidates:
                raise FileNotFoundError(f"no bundles under {bundles_root()}")
            root = candidates[-1]
        payload = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        horizons = {
            int(h): HorizonArtifacts.from_dict(a) for h, a in payload["horizons"].items()
        }
        bundle = cls(
            version=payload["version"],
            horizons=horizons,
            train_years=payload["train_years"],
            calibration_year=payload["calibration_year"],
            test_year=payload["test_year"],
            risk_class_names=payload["risk_class_names"],
            risk_quantiles=payload["risk_quantiles"],
            created_at=payload.get("created_at", ""),
            notes=payload.get("notes", {}),
            root=root,
        )
        return bundle

    def booster(self, horizon: int):
        from prometheus.models.lgbm import load_model

        artifacts = self.horizons[horizon]
        root = self.root or (bundles_root() / self.version)
        return load_model(root / artifacts.model_file)


def next_version(prefix: str = "v") -> str:
    existing = [p.name for p in bundles_root().iterdir() if p.is_dir()]
    numbers = [
        int(name[len(prefix) :]) for name in existing
        if name.startswith(prefix) and name[len(prefix) :].isdigit()
    ]
    return f"{prefix}{max(numbers, default=0) + 1}"


__all__ = ["MANIFEST", "HorizonArtifacts", "ModelBundle", "bundles_root", "next_version"]
