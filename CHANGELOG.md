# Changelog

## v0.1.5 (2026-03-22) [unreleased]
- Add quickstart notebook and Colab badge
- fix: use plain string license field for universal setuptools compatibility
- docs: add Limitations section covering known operational constraints
- fix: use importlib.metadata for __version__ (prevents drift from pyproject.toml)
- Add pytest to dev dependencies — fixes test collection in isolated venv
- Fix licence badge (BSD-3 -> MIT); remove emoji from discussion CTA
- fix: add shadow mode caveat to hit rate table + correct PS21/5 reference

## v0.1.5 (2026-03-21)
- Add cross-links to related libraries in README
- docs: replace pip install with uv add in README
- Add community CTA to README
- Add MIT license
- docs: add Benchmark Results section to README
- fix: QA audit fixes — v0.1.5
- Fix P0-P2 bugs: bootstrap p-value, shadow_model, power analysis, routing precision
- Rename Why bother → Performance for README consistency
- Add benchmark: champion/challenger governance vs manual tracking
- fix: remove scipy<1.11 upper bound — incompatible with Python 3.12
- fix: scale down populated_logger fixture to prevent test hangs on ARM
- Polish flagship README: badges, benchmark table, problem statement
- docs: add Databricks notebook link
- Add Databricks notebook link to README
- Add worked example link to README
- Fix: add to_polars()/to_pandas() to QuoteLogger; relax scipy/pandas constraints
- Add Related Libraries section to README
- fix: update cross-references to consolidated repos
- docs: add Performance section with benchmark summary

