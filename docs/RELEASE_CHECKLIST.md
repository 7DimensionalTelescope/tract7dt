# Docs Release Checklist

Use this checklist before pushing docs updates and publishing.

## Local validation

- [ ] Install docs dependencies:
  - `pip install -r docs/requirements.txt`
- [ ] Local preview builds cleanly:
  - `mkdocs serve`
- [ ] Optional static build passes:
  - `mkdocs build`
- [ ] Navigation renders all pages.
- [ ] Search returns expected key terms (config keys, commands, output columns).

## Content consistency

- [ ] `README.md` and docs agree on installation and quick-start steps.
- [ ] New/changed config keys are reflected in:
  - `tract7dt/data/sample_config.yaml`
  - `tract7dt/config.py` defaults
  - `docs/configuration.md` (and mapping page if needed)
- [ ] Behavior changes are documented in:
  - `docs/pipeline-behavior.md`
  - `docs/outputs.md`
  - `docs/performance.md`
- [ ] New output columns are documented.

## GitHub Pages setup

- [ ] `.github/workflows/docs.yml` is present and valid.
- [ ] `mkdocs.yml` is present and valid.
- [ ] `mkdocs.yml` has correct `site_url` for GitHub Pages.
- [ ] `docs/requirements.txt` includes required MkDocs packages.
- [ ] GitHub repository Pages source is set to **GitHub Actions**.

## Repository polish

- [ ] `README.md` includes docs pointer (or live URL once available).
- [ ] No stale/dead links in docs pages.
- [ ] Build artifacts are ignored (`site/`, `dist/`, `build/`, `*.egg-info/`).
- [ ] Commit message clearly indicates docs scope.

## Post-publish checks

- [ ] GitHub Actions docs workflow succeeded on default branch.
- [ ] Site URL works (`https://seoldh99.github.io/tract7dt/`).
- [ ] Search works on hosted site.
- [ ] Key pages verified:
  - Home
  - Configuration
  - Pipeline Behavior
  - Outputs
  - Troubleshooting
