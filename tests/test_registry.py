"""Tests for ModelRegistry."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from insurance_deploy import ModelRegistry, ModelVersion
from .conftest import DummyModel


class TestModelRegistryBasics:

    def test_register_returns_model_version(self, registry, champion_model):
        mv = registry.register(champion_model, "motor", "1.0", {"x": 1})
        assert isinstance(mv, ModelVersion)
        assert mv.name == "motor"
        assert mv.version == "1.0"
        assert mv.version_id == "motor:1.0"

    def test_version_id_format(self, registry, champion_model):
        mv = registry.register(champion_model, "home", "2024-Q1", {})
        assert mv.version_id == "home:2024-Q1"

    def test_register_stores_metadata(self, registry, champion_model):
        meta = {"training_date": "2024-01-01", "features": ["age", "ncd"], "auc": 0.82}
        mv = registry.register(champion_model, "motor", "1.0", meta)
        assert mv.metadata["training_date"] == "2024-01-01"
        assert mv.metadata["auc"] == 0.82

    def test_register_null_metadata_defaults_to_empty_dict(self, registry, champion_model):
        mv = registry.register(champion_model, "motor", "1.0", metadata=None)
        assert mv.metadata == {}

    def test_register_sets_registered_at(self, registry, champion_model):
        mv = registry.register(champion_model, "motor", "1.0")
        assert mv.registered_at is not None
        assert "T" in mv.registered_at  # ISO 8601

    def test_register_creates_model_file(self, registry, champion_model, tmp_dir):
        mv = registry.register(champion_model, "motor", "1.0")
        assert Path(mv.model_path).exists()

    def test_register_computes_hash(self, registry, champion_model):
        mv = registry.register(champion_model, "motor", "1.0")
        actual_hash = hashlib.sha256(Path(mv.model_path).read_bytes()).hexdigest()
        assert mv.model_hash == actual_hash

    def test_register_duplicate_raises(self, registry, champion_model):
        registry.register(champion_model, "motor", "1.0")
        with pytest.raises(ValueError, match="already registered"):
            registry.register(champion_model, "motor", "1.0")


class TestModelRegistryGet:

    def test_get_returns_correct_version(self, registry, champion_model, challenger_model):
        registry.register(champion_model, "motor", "1.0")
        registry.register(challenger_model, "motor", "2.0")
        mv = registry.get("motor", "1.0")
        assert mv.version == "1.0"

    def test_get_missing_raises(self, registry):
        with pytest.raises(KeyError):
            registry.get("motor", "99.0")

    def test_get_loads_model(self, registry, champion_model):
        import numpy as np
        mv = registry.register(champion_model, "motor", "1.0")
        # Clear cached model
        mv._model = None
        loaded = mv.model
        result = loaded.predict([[1, 2, 3]])
        assert result[0] == pytest.approx(400.0)


class TestModelRegistryList:

    def test_list_all(self, registry, champion_model, challenger_model):
        registry.register(champion_model, "motor", "1.0")
        registry.register(challenger_model, "motor", "2.0")
        registry.register(DummyModel(), "home", "1.0")
        versions = registry.list()
        assert len(versions) == 3

    def test_list_filtered_by_name(self, registry, champion_model, challenger_model):
        registry.register(champion_model, "motor", "1.0")
        registry.register(challenger_model, "motor", "2.0")
        registry.register(DummyModel(), "home", "1.0")
        motor = registry.list(name="motor")
        assert len(motor) == 2
        assert all(mv.name == "motor" for mv in motor)

    def test_list_sorted_by_registration_time(self, registry):
        for v in ["1.0", "2.0", "3.0"]:
            registry.register(DummyModel(), "motor", v)
        versions = registry.list("motor")
        assert [mv.version for mv in versions] == ["1.0", "2.0", "3.0"]


class TestModelRegistryChampion:

    def test_champion_returns_most_recent_by_default(self, registry):
        registry.register(DummyModel(), "motor", "1.0")
        registry.register(DummyModel(), "motor", "2.0")
        champ = registry.champion("motor")
        assert champ.version == "2.0"

    def test_set_champion_updates_designation(self, registry):
        registry.register(DummyModel(), "motor", "1.0")
        registry.register(DummyModel(), "motor", "2.0")
        registry.set_champion("motor", "1.0")
        champ = registry.champion("motor")
        assert champ.version == "1.0"

    def test_set_champion_marks_is_champion(self, registry):
        registry.register(DummyModel(), "motor", "1.0")
        registry.register(DummyModel(), "motor", "2.0")
        registry.set_champion("motor", "1.0")
        mv1 = registry.get("motor", "1.0")
        mv2 = registry.get("motor", "2.0")
        assert mv1.is_champion is True
        assert mv2.is_champion is False

    def test_champion_missing_name_raises(self, registry):
        with pytest.raises(KeyError):
            registry.champion("nonexistent")

    def test_set_champion_missing_version_raises(self, registry):
        registry.register(DummyModel(), "motor", "1.0")
        with pytest.raises(KeyError):
            registry.set_champion("motor", "99.0")


class TestModelRegistryPersistence:

    def test_registry_persists_across_instances(self, tmp_dir):
        path = tmp_dir / "registry"
        r1 = ModelRegistry(path)
        r1.register(DummyModel(), "motor", "1.0", {"tag": "v1"})
        r1.register(DummyModel(), "motor", "2.0", {"tag": "v2"})

        r2 = ModelRegistry(path)
        versions = r2.list("motor")
        assert len(versions) == 2
        assert versions[0].metadata["tag"] == "v1"

    def test_champion_designation_persists(self, tmp_dir):
        path = tmp_dir / "registry"
        r1 = ModelRegistry(path)
        r1.register(DummyModel(), "motor", "1.0")
        r1.register(DummyModel(), "motor", "2.0")
        r1.set_champion("motor", "1.0")

        r2 = ModelRegistry(path)
        assert r2.champion("motor").version == "1.0"

    def test_hash_verification_on_load(self, tmp_dir):
        path = tmp_dir / "registry"
        r1 = ModelRegistry(path)
        mv = r1.register(DummyModel(), "motor", "1.0")

        # Tamper with the model file
        Path(mv.model_path).write_bytes(b"corrupted data")

        r2 = ModelRegistry(path)
        loaded_mv = r2.get("motor", "1.0")
        with pytest.raises(ValueError, match="hash mismatch"):
            _ = loaded_mv.model
