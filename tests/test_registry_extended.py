"""
Extended tests for ModelRegistry and ModelVersion.

Gaps covered:
- register duplicate version raises ValueError
- get nonexistent version raises KeyError
- champion() raises KeyError when no versions registered for name
- list() with and without name filter
- list() sorted by registered_at
- set_champion() explicitly promotes a version
- set_champion() for nonexistent version raises KeyError
- ModelVersion.predict() delegates to model
- ModelVersion.__repr__ shows champion marker
- Registry persistence: data survives across instances
- Hash verification: tampering detected
- Metadata stored and retrieved correctly
- is_champion flag correct after set_champion()
- Multiple model families in same registry
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from insurance_deploy import ModelRegistry, ModelVersion


# ---------------------------------------------------------------------------
# Minimal test model
# ---------------------------------------------------------------------------

class DummyPredictor:
    def __init__(self, value=1.0):
        self.value = value

    def predict(self, X):
        import numpy as np
        return np.full(len(X) if hasattr(X, "__len__") else 1, self.value)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def registry(tmp_dir):
    return ModelRegistry(tmp_dir / "reg")


# ---------------------------------------------------------------------------
# Basic registration
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_register_returns_model_version(self, registry):
        mv = registry.register(DummyPredictor(), name="motor", version="1.0")
        assert isinstance(mv, ModelVersion)

    def test_version_id_format(self, registry):
        mv = registry.register(DummyPredictor(), name="motor", version="2.0")
        assert mv.version_id == "motor:2.0"

    def test_name_and_version_stored(self, registry):
        mv = registry.register(DummyPredictor(), name="home", version="1.0")
        assert mv.name == "home"
        assert mv.version == "1.0"

    def test_metadata_stored(self, registry):
        meta = {"training_date": "2024-01-01", "features": ["age", "ncd"]}
        mv = registry.register(DummyPredictor(), name="motor", version="1.0", metadata=meta)
        assert mv.metadata["training_date"] == "2024-01-01"
        assert mv.metadata["features"] == ["age", "ncd"]

    def test_empty_metadata_default(self, registry):
        mv = registry.register(DummyPredictor(), name="motor", version="1.0")
        assert mv.metadata == {}

    def test_duplicate_version_raises(self, registry):
        registry.register(DummyPredictor(), name="motor", version="1.0")
        with pytest.raises(ValueError, match="already registered"):
            registry.register(DummyPredictor(), name="motor", version="1.0")

    def test_same_version_different_name_ok(self, registry):
        """Same version string under different model names should be fine."""
        registry.register(DummyPredictor(), name="motor", version="1.0")
        registry.register(DummyPredictor(), name="home", version="1.0")
        assert len(registry.list()) == 2

    def test_model_hash_is_sha256_hex(self, registry):
        mv = registry.register(DummyPredictor(), name="motor", version="1.0")
        assert len(mv.model_hash) == 64  # SHA-256 = 64 hex chars
        assert all(c in "0123456789abcdef" for c in mv.model_hash)

    def test_model_path_exists(self, registry):
        mv = registry.register(DummyPredictor(), name="motor", version="1.0")
        assert Path(mv.model_path).exists()


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------

class TestGet:

    def test_get_returns_correct_version(self, registry):
        mv1 = registry.register(DummyPredictor(1.0), name="motor", version="1.0")
        mv2 = registry.register(DummyPredictor(2.0), name="motor", version="2.0")
        retrieved = registry.get("motor", "1.0")
        assert retrieved.version_id == "motor:1.0"

    def test_get_nonexistent_raises_key_error(self, registry):
        with pytest.raises(KeyError, match="not found"):
            registry.get("motor", "9.9")

    def test_get_after_register_returns_same_id(self, registry):
        mv = registry.register(DummyPredictor(), name="motor", version="1.0")
        retrieved = registry.get("motor", "1.0")
        assert retrieved.version_id == mv.version_id


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------

class TestList:

    def test_list_all(self, registry):
        registry.register(DummyPredictor(), name="motor", version="1.0")
        registry.register(DummyPredictor(), name="motor", version="2.0")
        registry.register(DummyPredictor(), name="home", version="1.0")
        versions = registry.list()
        assert len(versions) == 3

    def test_list_filtered_by_name(self, registry):
        registry.register(DummyPredictor(), name="motor", version="1.0")
        registry.register(DummyPredictor(), name="motor", version="2.0")
        registry.register(DummyPredictor(), name="home", version="1.0")
        motor_versions = registry.list("motor")
        assert len(motor_versions) == 2
        assert all(mv.name == "motor" for mv in motor_versions)

    def test_list_empty_for_unknown_name(self, registry):
        registry.register(DummyPredictor(), name="motor", version="1.0")
        unknown = registry.list("property")
        assert unknown == []

    def test_list_empty_registry(self, registry):
        assert registry.list() == []

    def test_list_sorted_by_registered_at(self, registry):
        registry.register(DummyPredictor(), name="motor", version="1.0")
        registry.register(DummyPredictor(), name="motor", version="2.0")
        versions = registry.list("motor")
        timestamps = [mv.registered_at for mv in versions]
        assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# Champion
# ---------------------------------------------------------------------------

class TestChampion:

    def test_champion_returns_latest_by_default(self, registry):
        registry.register(DummyPredictor(), name="motor", version="1.0")
        registry.register(DummyPredictor(), name="motor", version="2.0")
        champ = registry.champion("motor")
        # Latest registered should be returned (version 2.0 registered after 1.0)
        assert champ.version == "2.0"

    def test_champion_no_versions_raises(self, registry):
        with pytest.raises(KeyError):
            registry.champion("nonexistent")

    def test_set_champion_explicit(self, registry):
        registry.register(DummyPredictor(), name="motor", version="1.0")
        registry.register(DummyPredictor(), name="motor", version="2.0")
        registry.set_champion("motor", "1.0")
        champ = registry.champion("motor")
        assert champ.version == "1.0"

    def test_set_champion_updates_is_champion_flag(self, registry):
        registry.register(DummyPredictor(), name="motor", version="1.0")
        registry.register(DummyPredictor(), name="motor", version="2.0")
        registry.set_champion("motor", "1.0")
        mv1 = registry.get("motor", "1.0")
        mv2 = registry.get("motor", "2.0")
        assert mv1.is_champion is True
        assert mv2.is_champion is False

    def test_set_champion_nonexistent_raises(self, registry):
        registry.register(DummyPredictor(), name="motor", version="1.0")
        with pytest.raises(KeyError):
            registry.set_champion("motor", "9.9")

    def test_set_champion_returns_model_version(self, registry):
        registry.register(DummyPredictor(), name="motor", version="1.0")
        mv = registry.set_champion("motor", "1.0")
        assert isinstance(mv, ModelVersion)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:

    def test_registry_survives_reload(self, tmp_dir):
        reg_path = tmp_dir / "reg"
        reg1 = ModelRegistry(reg_path)
        reg1.register(DummyPredictor(), name="motor", version="1.0",
                      metadata={"train": "2024-01-01"})

        reg2 = ModelRegistry(reg_path)
        mv = reg2.get("motor", "1.0")
        assert mv.version_id == "motor:1.0"
        assert mv.metadata["train"] == "2024-01-01"

    def test_champion_persists_across_reload(self, tmp_dir):
        reg_path = tmp_dir / "reg"
        reg1 = ModelRegistry(reg_path)
        reg1.register(DummyPredictor(), name="motor", version="1.0")
        reg1.register(DummyPredictor(), name="motor", version="2.0")
        reg1.set_champion("motor", "1.0")

        reg2 = ModelRegistry(reg_path)
        champ = reg2.champion("motor")
        assert champ.version == "1.0"

    def test_all_versions_persist(self, tmp_dir):
        reg_path = tmp_dir / "reg"
        reg1 = ModelRegistry(reg_path)
        for v in ["1.0", "2.0", "3.0"]:
            reg1.register(DummyPredictor(), name="motor", version=v)

        reg2 = ModelRegistry(reg_path)
        assert len(reg2.list("motor")) == 3


# ---------------------------------------------------------------------------
# ModelVersion predict and model access
# ---------------------------------------------------------------------------

class TestModelVersionPredict:

    def test_predict_delegates_to_model(self, registry):
        import numpy as np
        mv = registry.register(DummyPredictor(value=99.0), name="motor", version="1.0")
        result = mv.predict(np.zeros(5))
        np.testing.assert_array_equal(result, np.full(5, 99.0))

    def test_model_property_loads_and_verifies(self, registry):
        mv = registry.register(DummyPredictor(42.0), name="motor", version="1.0")
        # Force lazy load by accessing .model (the fixture already has it cached)
        # Clear the cache to test load path
        mv._model = None
        loaded = mv.model
        assert loaded is not None

    def test_model_hash_mismatch_raises(self, registry):
        import hashlib
        mv = registry.register(DummyPredictor(), name="motor", version="1.0")
        mv._model = None  # clear cache to force disk load
        # Corrupt the stored hash
        mv.model_hash = "a" * 64
        with pytest.raises(ValueError, match="hash mismatch"):
            _ = mv.model


# ---------------------------------------------------------------------------
# ModelVersion repr
# ---------------------------------------------------------------------------

class TestModelVersionRepr:

    def test_repr_unflagged(self, registry):
        mv = registry.register(DummyPredictor(), name="motor", version="1.0")
        r = repr(mv)
        assert "motor:1.0" in r
        assert "[champion]" not in r

    def test_repr_champion_flagged(self, registry):
        registry.register(DummyPredictor(), name="motor", version="1.0")
        mv = registry.set_champion("motor", "1.0")
        r = repr(mv)
        assert "[champion]" in r

    def test_repr_shows_registered_date(self, registry):
        mv = registry.register(DummyPredictor(), name="motor", version="1.0")
        r = repr(mv)
        # Should contain at least the year
        assert "202" in r
