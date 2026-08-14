# Live Inference Testing

The app's local sentiment and emotion pipelines (`load_sentiment_pipeline`/`load_emotion_pipeline` in [streamlit_app.py](../src/streamlit_app.py)) have a pytest suite under [`tests/`](../tests/) that runs real, unmocked inference against the pinned models — no mocking of `transformers`, no fake responses. A GitHub Actions workflow ([`.github/workflows/tests.yml`](../.github/workflows/tests.yml)) runs the suite on every push and PR.

## What's tested, and what isn't

| | `predict_sentiment` / `predict_emotion` | `get_llm_response` |
|---|---|---|
| Covered by these tests? | Yes | No |
| Why | Local `transformers.pipeline`, no token, no network, no billing (see the table in [hf-token-handling.md](./hf-token-handling.md#transformers-api-vs-inference-providers-api)) | Needs `HF_TOKEN` + a network call to HF's Inference Providers API, and can legitimately 402 on billing/credit exhaustion — not something CI should depend on |

`tests/test_sentiment_analysis.py` and `tests/test_emotion_analysis.py` each run a small set of parametrized, strong-signal inputs (e.g. "I absolutely love this, best day of my life!" → `positive`) through the real model and assert on the returned label and confidence — this is a smoke test that the pinned model still behaves as expected, not an attempt to exhaustively validate model accuracy.

## Pinning

`SENTIMENT_MODEL` and `EMOTION_MODEL` are module-level constants in `streamlit_app.py`:

```python
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
```

The app's loader functions and the test suite both reference these constants, so there's a single source of truth — no model ID string is duplicated between app code and tests. Each test file also has a `test_*_model_is_pinned` assertion that fails loudly if either constant ever changes unexpectedly.

Note this pins the **model repo**, not a specific weights revision — `pipeline(model=...)` without a `revision=` resolves to whatever the repo's default branch currently points to. If the upstream authors push new weights, a fresh download (e.g. after a cache miss) could pull different weights than a previous run, with this app's code unchanged. Full reproducibility would mean adding `revision="<commit-sha>"` to both `pipeline()` calls — not done here, since it also means the model never auto-updates and needs manual maintenance. Worth revisiting if these models ever exhibit surprising drift.

## Loading once per test run

`tests/conftest.py` has a single session-scoped fixture:

```python
@pytest.fixture(scope="session")
def app_module():
    import streamlit_app
    return streamlit_app
```

The mechanism that actually avoids reloading the models for every test is plain Python, not this fixture and not Streamlit: `import streamlit_app` triggers the module's own eager, module-level pipeline loads —

```python
analyzer = load_sentiment_pipeline()
emotion_analyzer = load_emotion_pipeline()
```

— and Python caches imported modules in `sys.modules`, so this only executes once per test process no matter how many test files request `app_module`. `@st.cache_resource` (already on both loader functions, for the live Streamlit app's benefit) is a redundant second safety net in the test path, not what's doing the work — worth stating plainly here so it isn't overclaimed.

## The `__main__` guard

`streamlit_app.py` has no other entry point — `streamlit run` executes it as a script with `__name__ == "__main__"`. The entire interactive section (chat UI, `st.chat_input`, session state, sidebar token input) is wrapped in `if __name__ == "__main__":`, so a plain `import streamlit_app` (what the tests do) only defines functions/constants and runs the two eager pipeline loads — it never touches any Streamlit widget API. This is why the tests don't need `streamlit.testing.v1.AppTest` or any Streamlit-specific test harness: they import the module like any other Python module and call `predict_sentiment`/`predict_emotion` directly. It also avoids relying on Streamlit's "missing ScriptRunContext, ignorable in bare mode" fallback behavior, which is an internal diagnostic affordance, not a stable, documented contract to build tests on.

## Running locally

```bash
cd chatbot_ai_v2
pip install -r requirements-test.txt
pytest -v
```

`requirements-test.txt` layers `pytest` on top of `requirements.txt` (`-r requirements.txt`) rather than adding it to `requirements.txt` directly, so the Docker deploy image (`Dockerfile`, `pip3 install -r requirements.txt`) doesn't gain a test dependency it'll never use at runtime.

## CI workflow (`.github/workflows/tests.yml`)

Runs on every push (any branch) and on pull requests targeting `main`. No secrets are needed — see the table above, these tests never touch `HF_TOKEN` or the network beyond downloading model weights from the Hugging Face Hub.

**Python 3.13**, matching `Dockerfile`'s deployment target (`python:3.13.5-slim`) — not `.python-version`'s `3.8.1`, which is a local-dev-only pin for Intel-Mac `torch` wheel availability (see [README.md](../README.md)) and irrelevant to GitHub's Linux runners.

**CPU-only torch.** `pip install torch` on PyPI pulls CUDA-bundled wheels by default, even on GPU-less `ubuntu-latest` runners — it still works (falls back to CPU transparently) but wastes significant download time on every cache-miss run. The workflow installs from the CPU wheel index first, so the default resolver has nothing left to do:

```yaml
- name: Install CPU-only torch
  run: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Model weight caching.** `transformers`/`huggingface_hub` cache downloaded weights under `HF_HOME` (default `~/.cache/huggingface`). The workflow sets `HF_HOME` explicitly to a path inside `${{ github.workspace }}`, as a job-level `env:`, so the cache location is visible in the workflow file rather than an implicit default, and the cache step references `${{ env.HF_HOME }}` instead of repeating the literal path:

```yaml
env:
  HF_HOME: ${{ github.workspace }}/.hf_cache
steps:
  - name: Cache Hugging Face model weights
    uses: actions/cache@v4
    with:
      path: ${{ env.HF_HOME }}
      key: hf-models-${{ runner.os }}-cardiffnlp-twitter-roberta-base-sentiment-latest-j-hartmann-emotion-english-distilroberta-base
      restore-keys: |
        hf-models-${{ runner.os }}-
```

How this avoids re-downloading weights on every run: `actions/cache` first looks for an *exact* match on `key`. On a hit, `${{ env.HF_HOME }}` is restored before the test step runs, so `transformers` finds the weights already on disk and makes zero network calls. On a miss (first run ever, or the key changed), `restore-keys` falls back to a *prefix* match — the most recently-created cache whose key starts with `hf-models-${{ runner.os }}-` — giving a partially-warm cache instead of a fully cold one.

The model IDs are embedded directly in the key (slashes replaced with dashes) rather than an opaque version number, so the key is self-documenting and a diff to `SENTIMENT_MODEL`/`EMOTION_MODEL` naturally prompts updating it too. The key isn't computed at runtime from those constants (that would need an extra step installing deps and importing them before the cache step), so it can fall out of sync if someone forgets — but the failure mode is benign: HF's on-disk cache is content-addressed per model repo (`hub/models--org--name/...`), so a stale key just means the new model downloads fresh and sits alongside old cache entries, never wrong weights being used.

**Pip dependency caching** uses `actions/setup-python`'s built-in `cache: pip`, keyed off both `requirements.txt` and `requirements-test.txt` via `cache-dependency-path`.

## Why this workflow doesn't gate the HF Space deploy

`.github/workflows/sync-to-hub.yml` (see [github-actions-sync.md](./github-actions-sync.md)) is untouched by this change — it still deploys on every push to `main`, independent of `tests.yml`'s outcome. This is deliberate, not an oversight: gating deploy on this test suite would mean a transient failure unrelated to an actual app bug — a flaky download, a temporary Hugging Face Hub hiccup, an upstream model owner tweaking output slightly — could block deploying otherwise-working code. The two workflows run in parallel and are read independently from the repo's checks list.

## Changelog

**2026-08-14** — Added the live inference test suite and this CI workflow.

- `src/streamlit_app.py`: extracted `SENTIMENT_MODEL`/`EMOTION_MODEL` constants (previously inline string literals inside `load_sentiment_pipeline`/`load_emotion_pipeline`); wrapped the interactive UI section (everything from `st.title(...)` onward) in `if __name__ == "__main__":` so the module can be imported for testing without executing any Streamlit widget calls. No behavior change under `streamlit run`.
- Added `tests/conftest.py`, `tests/test_sentiment_analysis.py`, `tests/test_emotion_analysis.py`, and `pytest.ini` (`testpaths = tests`, `pythonpath = src`).
- Added `requirements-test.txt` (layers `pytest` on top of `requirements.txt`) so the Docker deploy image doesn't pick up a test-only dependency.
- Added `.github/workflows/tests.yml`: runs the suite on every push and PR to `main`, with CPU-only torch, `HF_HOME`-based model weight caching (`actions/cache`, keyed on the pinned model IDs with a prefix-based `restore-keys` fallback), and pip caching via `actions/setup-python`. Pinned to Python 3.13 to match `Dockerfile`'s deployment target rather than `.python-version`'s local-dev 3.8 pin.
- Deliberately did **not** modify `sync-to-hub.yml` or gate deploy on tests passing — the two workflows are intentionally independent, so upstream flakiness can't block a deploy of working app code. See "Why this workflow doesn't gate the HF Space deploy" above.
