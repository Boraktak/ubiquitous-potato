# CI workflow

`github-actions-ci.yml` is the GitHub Actions workflow for this repo. It is
parked here instead of `.github/workflows/` because the bot account that opened
the pull request does not hold the `workflows` permission, so GitHub rejects any
push that adds or edits a workflow file.

To activate it, a human with write access runs:

```bash
mkdir -p .github/workflows
git mv ci/github-actions-ci.yml .github/workflows/ci.yml
git commit -m "Enable GitHub Actions CI"
git push
```

The workflow runs `pytest -q` and then `python dialectic.py` on Python 3.10,
3.11 and 3.12 using CPU-only torch wheels.

## Running the tests locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest -q
```
