"""Tests for Experiment routing and lifecycle."""

from __future__ import annotations

import hashlib
import warnings

import pytest

from insurance_deploy import Experiment
from .conftest import DummyModel


class TestExperimentRouting:

    def test_route_returns_champion_or_challenger(self, experiment):
        arm = experiment.route("POL-001")
        assert arm in ("champion", "challenger")

    def test_route_is_deterministic(self, experiment):
        """Same policy_id must always return the same arm."""
        policy_ids = [f"POL-{i:05d}" for i in range(200)]
        results1 = [experiment.route(pid) for pid in policy_ids]
        results2 = [experiment.route(pid) for pid in policy_ids]
        assert results1 == results2

    def test_route_is_deterministic_across_instances(self, champion_mv, challenger_mv):
        """Two Experiment instances with the same name must route identically."""
        kwargs = dict(
            name="test_exp",
            champion=champion_mv,
            challenger=challenger_mv,
            challenger_pct=0.20,
        )
        exp1 = Experiment(**kwargs)
        exp2 = Experiment(**kwargs)
        pids = [f"POLICY-{i}" for i in range(100)]
        assert [exp1.route(p) for p in pids] == [exp2.route(p) for p in pids]

    def test_route_challenger_fraction_approximately_correct(self, champion_mv, challenger_mv):
        """Routing should produce approximately challenger_pct challenger assignments."""
        exp = Experiment(
            "fraction_test", champion_mv, challenger_mv, challenger_pct=0.20
        )
        pids = [f"X-{i}" for i in range(10_000)]
        challenger_count = sum(1 for p in pids if exp.route(p) == "challenger")
        fraction = challenger_count / len(pids)
        # Should be close to 20% (within 1pp for 10k samples)
        assert 0.19 <= fraction <= 0.21

    def test_route_default_10pct_split(self, champion_mv, challenger_mv):
        exp = Experiment("default_split", champion_mv, challenger_mv)
        pids = [f"P-{i}" for i in range(10_000)]
        challenger_count = sum(1 for p in pids if exp.route(p) == "challenger")
        fraction = challenger_count / len(pids)
        assert 0.09 <= fraction <= 0.11

    def test_route_uses_sha256_algorithm(self, experiment):
        """Verify routing matches expected SHA-256 computation."""
        policy_id = "POL-KNOWN-001"
        key = (policy_id + experiment.name).encode()
        digest = hashlib.sha256(key).hexdigest()
        slot = int(digest[-8:], 16) % 100
        threshold = int(experiment.challenger_pct * 100)
        expected = "challenger" if slot < threshold else "champion"
        assert experiment.route(policy_id) == expected

    def test_different_experiment_names_produce_different_routing(
        self, champion_mv, challenger_mv
    ):
        exp1 = Experiment("exp_alpha", champion_mv, challenger_mv, challenger_pct=0.50)
        exp2 = Experiment("exp_beta", champion_mv, challenger_mv, challenger_pct=0.50)
        pids = [f"P-{i}" for i in range(1000)]
        routes1 = [exp1.route(p) for p in pids]
        routes2 = [exp2.route(p) for p in pids]
        # Different names should produce different (but not completely opposite) routing
        assert routes1 != routes2

    def test_live_model_returns_champion_in_shadow_mode(self, experiment):
        """In shadow mode, live_model is always champion."""
        pids = [f"P-{i}" for i in range(50)]
        for pid in pids:
            assert experiment.live_model(pid).version_id == experiment.champion.version_id

    def test_live_model_follows_route_in_live_mode(self, champion_mv, challenger_mv):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = Experiment(
                "live_test", champion_mv, challenger_mv,
                challenger_pct=0.50, mode="live"
            )
        pids = [f"P-{i}" for i in range(100)]
        for pid in pids:
            arm = exp.route(pid)
            model = exp.live_model(pid)
            if arm == "challenger":
                assert model.version_id == challenger_mv.version_id
            else:
                assert model.version_id == champion_mv.version_id


class TestExperimentValidation:

    def test_invalid_challenger_pct_zero_raises(self, champion_mv, challenger_mv):
        with pytest.raises(ValueError, match="challenger_pct"):
            Experiment("test", champion_mv, challenger_mv, challenger_pct=0.0)

    def test_invalid_challenger_pct_one_raises(self, champion_mv, challenger_mv):
        with pytest.raises(ValueError, match="challenger_pct"):
            Experiment("test", champion_mv, challenger_mv, challenger_pct=1.0)

    def test_invalid_challenger_pct_negative_raises(self, champion_mv, challenger_mv):
        with pytest.raises(ValueError, match="challenger_pct"):
            Experiment("test", champion_mv, challenger_mv, challenger_pct=-0.1)

    def test_invalid_mode_raises(self, champion_mv, challenger_mv):
        with pytest.raises(ValueError, match="mode"):
            Experiment("test", champion_mv, challenger_mv, mode="bandit")

    def test_live_mode_emits_warning(self, champion_mv, challenger_mv):
        with pytest.warns(UserWarning, match="Consumer Duty"):
            Experiment("test", champion_mv, challenger_mv, mode="live")


class TestExperimentLifecycle:

    def test_new_experiment_is_active(self, experiment):
        assert experiment.is_active() is True

    def test_deactivate_sets_inactive(self, experiment):
        experiment.deactivate()
        assert experiment.is_active() is False

    def test_route_on_deactivated_raises(self, experiment):
        experiment.deactivate()
        with pytest.raises(RuntimeError, match="deactivated"):
            experiment.route("POL-001")

    def test_deactivated_at_set_on_deactivate(self, experiment):
        assert experiment.deactivated_at is None
        experiment.deactivate()
        assert experiment.deactivated_at is not None

    def test_repr_shows_status(self, experiment):
        r = repr(experiment)
        assert "active" in r
        experiment.deactivate()
        r2 = repr(experiment)
        assert "deactivated" in r2

    def test_repr_shows_mode_and_split(self, experiment):
        r = repr(experiment)
        assert "shadow" in r
        assert "10%" in r
