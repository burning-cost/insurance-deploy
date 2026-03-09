"""
ModelRegistry — version-tagged storage for pricing model objects.

Design choices
--------------
Storage is local filesystem: JSON metadata alongside joblib-serialised model
objects. No database, no cloud dependencies. Most UK pricing teams do not
operate cloud-native infrastructure for pricing tooling — they run Python on a
laptop or Databricks notebook, not in Kubernetes.

The registry is append-only. You cannot delete or overwrite a registered model
version. This is intentional: the audit trail requires knowing what was
deployed, not just what is deployed now.

Hash verification: each registered model gets a SHA-256 hash of its serialised
bytes. Load-time verification detects file corruption or tampering.

Security note
-------------
Model objects are stored as joblib files (pickle-based). Loading a registry
file from an untrusted source is dangerous — pickle can execute arbitrary code
on load. Trust your registry directory as you would trust executable code.
"""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import joblib


@dataclass
class ModelVersion:
    """A registered model with version metadata."""

    name: str
    version: str
    version_id: str  # "{name}:{version}"
    registered_at: str  # ISO 8601
    metadata: dict[str, Any]
    model_hash: str  # SHA-256 of joblib bytes
    model_path: str  # absolute path to .joblib file
    is_champion: bool = False
    _model: Any = field(default=None, repr=False)

    @property
    def model(self) -> Any:
        """Load and return the model object, verifying its hash."""
        if self._model is None:
            self._model = _load_and_verify(self.model_path, self.model_hash)
        return self._model

    def predict(self, X: Any) -> Any:
        """Convenience wrapper: call model.predict(X)."""
        return self.model.predict(X)

    def __repr__(self) -> str:
        champion_marker = " [champion]" if self.is_champion else ""
        return (
            f"ModelVersion('{self.version_id}'{champion_marker}, "
            f"registered={self.registered_at[:10]})"
        )


def _serialise_and_hash(model: Any, path: Path) -> str:
    """Serialise model to path, return SHA-256 hex digest."""
    joblib.dump(model, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return sha


def _load_and_verify(path: str, expected_hash: str) -> Any:
    """Load model from path, verify SHA-256 hash."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != expected_hash:
        raise ValueError(
            f"Model hash mismatch for {path}.\n"
            f"Expected: {expected_hash}\n"
            f"  Actual: {actual}\n"
            "The file may be corrupted or tampered with."
        )
    return joblib.load(path)


class ModelRegistry:
    """
    Append-only registry for pricing model versions.

    Parameters
    ----------
    path : str or Path
        Directory for storing model files and metadata. Created if absent.

    Examples
    --------
    >>> registry = ModelRegistry("./registry")
    >>> mv = registry.register(
    ...     model, name="motor_v3", version="1.0",
    ...     metadata={"training_date": "2024-01-01", "features": ["age", "ncd"]}
    ... )
    >>> champion = registry.champion("motor_v3")
    """

    def __init__(self, path: str | Path = "./model_registry") -> None:
        self.path = Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.path / "registry.json"
        self._versions: dict[str, ModelVersion] = {}
        self._champions: dict[str, str] = {}  # name -> version_id
        self._load_metadata()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        model: Any,
        name: str,
        version: str,
        metadata: dict[str, Any] | None = None,
    ) -> ModelVersion:
        """
        Register a new model version.

        Parameters
        ----------
        model : any
            Any object with a ``.predict()`` method (sklearn-compatible).
        name : str
            Model family name, e.g. ``"motor_v3"``.
        version : str
            Version string, e.g. ``"1.0"`` or ``"2024-Q1"``.
        metadata : dict, optional
            Arbitrary key/value store: training date, features list,
            validation KPIs, hyperparameters, etc.

        Returns
        -------
        ModelVersion

        Raises
        ------
        ValueError
            If this name+version combination is already registered.
        """
        version_id = f"{name}:{version}"
        if version_id in self._versions:
            raise ValueError(
                f"Model version '{version_id}' is already registered. "
                "The registry is append-only — create a new version string."
            )
        if metadata is None:
            metadata = {}

        model_path = self.path / f"{version_id.replace(':', '_')}.joblib"
        model_hash = _serialise_and_hash(model, model_path)

        mv = ModelVersion(
            name=name,
            version=version,
            version_id=version_id,
            registered_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
            model_hash=model_hash,
            model_path=str(model_path),
            is_champion=False,
            _model=model,
        )
        self._versions[version_id] = mv
        self._save_metadata()
        return mv

    def get(self, name: str, version: str) -> ModelVersion:
        """Retrieve a specific model version."""
        version_id = f"{name}:{version}"
        if version_id not in self._versions:
            raise KeyError(f"Model version '{version_id}' not found in registry.")
        return self._versions[version_id]

    def list(self, name: Optional[str] = None) -> list[ModelVersion]:
        """
        List registered model versions.

        Parameters
        ----------
        name : str, optional
            Filter by model family name. Returns all versions if omitted.
        """
        versions = list(self._versions.values())
        if name is not None:
            versions = [v for v in versions if v.name == name]
        return sorted(versions, key=lambda v: v.registered_at)

    def champion(self, name: str) -> ModelVersion:
        """
        Return the current champion for a model family.

        The most recently registered version is treated as champion unless
        explicitly set via ``set_champion()``.
        """
        if name in self._champions:
            return self._versions[self._champions[name]]
        # Default: most recently registered version with this name
        candidates = [v for v in self._versions.values() if v.name == name]
        if not candidates:
            raise KeyError(f"No model versions registered for '{name}'.")
        return max(candidates, key=lambda v: v.registered_at)

    def set_champion(self, name: str, version: str) -> ModelVersion:
        """
        Explicitly promote a version to champion.

        This does not demote the current champion — both versions remain in
        the registry. The champion designation is metadata only; it does not
        affect routing (that is Experiment's responsibility).
        """
        version_id = f"{name}:{version}"
        if version_id not in self._versions:
            raise KeyError(f"Model version '{version_id}' not found.")
        # Update is_champion flags
        for vid, mv in self._versions.items():
            if mv.name == name:
                mv.is_champion = False
        self._versions[version_id].is_champion = True
        self._champions[name] = version_id
        self._save_metadata()
        return self._versions[version_id]

    # ------------------------------------------------------------------
    # Internal persistence
    # ------------------------------------------------------------------

    def _save_metadata(self) -> None:
        records = []
        for mv in self._versions.values():
            rec = {
                "name": mv.name,
                "version": mv.version,
                "version_id": mv.version_id,
                "registered_at": mv.registered_at,
                "metadata": mv.metadata,
                "model_hash": mv.model_hash,
                "model_path": mv.model_path,
                "is_champion": mv.is_champion,
            }
            records.append(rec)
        payload = {"versions": records, "champions": self._champions}
        self._meta_path.write_text(json.dumps(payload, indent=2))

    def _load_metadata(self) -> None:
        if not self._meta_path.exists():
            return
        payload = json.loads(self._meta_path.read_text())
        for rec in payload.get("versions", []):
            mv = ModelVersion(
                name=rec["name"],
                version=rec["version"],
                version_id=rec["version_id"],
                registered_at=rec["registered_at"],
                metadata=rec["metadata"],
                model_hash=rec["model_hash"],
                model_path=rec["model_path"],
                is_champion=rec.get("is_champion", False),
                _model=None,
            )
            self._versions[mv.version_id] = mv
        self._champions = payload.get("champions", {})
