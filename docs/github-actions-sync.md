# GitHub → Hugging Face Sync

Pushes to `main` on GitHub ([sync-to-hub.yml](../.github/workflows/sync-to-hub.yml)) are
mirrored to the HF Space (`aricode00x/adaptive_ai_chatbot`) automatically, so GitHub can be the
repo you actually push to day-to-day instead of the `hf` remote.

This is a **git-to-git push** (`git push https://aricode00x:$HF_TOKEN@huggingface.co/spaces/aricode00x/adaptive_ai_chatbot main`),
not Hugging Face's `hub-sync` action — the whole commit history and object store go over,
not just a snapshot of file contents.

## One-time setup: add the `HF_TOKEN` GitHub secret

The workflow needs write access to the Space, via a GitHub Actions secret:

1. Create an HF access token at https://huggingface.co/settings/tokens with **write** access.
2. Add it as a repo secret named `HF_TOKEN`, at
   `github.com/ar-to/adaptive_ai_chatbot/settings/secrets/actions` → "New repository secret".
   (`gh` isn't installed locally; if you install and auth it, `gh secret set HF_TOKEN` works
   too.)

This `HF_TOKEN` is unrelated to the app's own `HF_TOKEN` covered in
[hf-token-handling.md](./hf-token-handling.md) — same name, two different places
(GitHub Actions secret vs. Space runtime secret / pasted session token) serving two different
purposes (pushing commits vs. calling the Inference Providers API). Setting one does not set
the other.

## Triggers

- Automatically, on every push to `main`.
- Manually, via the repo's **Actions** tab → "Sync to Hugging Face hub" → "Run workflow".

## Failure mode: diverged history

The push is a plain `git push` — no `--force`. If the Space's `main` has diverged from
GitHub's (e.g. someone edited a file directly in the HF web UI), the push is rejected and the
workflow run fails rather than silently overwriting Space history. To resolve: reconcile the
histories locally (e.g. `git fetch hf && git merge hf/main` or rebase), push the merge to
GitHub, and the next sync run will succeed.
