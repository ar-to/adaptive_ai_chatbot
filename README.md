---
title: Chatbot AI v2
emoji: 🚀
colorFrom: red
colorTo: red
sdk: docker
app_port: 8501
tags:
- streamlit
pinned: false
short_description: https://huggingface.co/docs/hub/en/spaces-sdks-streamlit
---

# Welcome to Streamlit!

Edit `/src/streamlit_app.py` to customize this app to your heart's desire. :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

# Returning

The venv for this project lives *outside* the repo, at `~/.venvs/chatbot_ai_v2` — this
keeps the local folder free of large `site-packages`/metadata to clean up. It's pinned to
**Python 3.8** because `torch` dropped Intel-Mac wheels after `2.2.2`, and 3.8 is the last
interpreter pip can still resolve that version for automatically.

```bash
cd chatbot_ai_v2

# one-time only, if ~/.venvs/chatbot_ai_v2 doesn't exist yet
python3.8 -m venv ~/.venvs/chatbot_ai_v2

# every time you return: activate the venv
source ~/.venvs/chatbot_ai_v2/bin/activate

# one-time per venv: install deps (pip resolves torch==2.2.2 + transformers==4.46.3
# automatically on Python 3.8 / Intel Mac — no manual pinning needed)
pip install -r requirements.txt

# run the app
streamlit run src/streamlit_app.py
```

Tip: add this to `~/.zshrc` to activate with one word instead of the full path:
```bash
activate() { source ~/.venvs/"$1"/bin/activate; }
```
then just run `activate chatbot_ai_v2`.

Deploying to the Space is unaffected by any of this — Hugging Face builds the `Dockerfile`,
which installs from `requirements.txt` directly on its own (Linux) infrastructure.

# Enabling real AI replies

By default this Space has no `HF_TOKEN` configured, so it runs on free local sentiment
analysis plus canned empathetic replies — $0 cost to the maintainer. To get real
LLM-generated replies:

- **Paste your own token**: enter a Hugging Face token in the sidebar. It's used for your
  session only and is never stored.
- **Or duplicate the Space**: click "Duplicate this Space," make your copy private, and add
  `HF_TOKEN` under Settings → Variables and secrets so it's always on.

Get a free token at https://huggingface.co/settings/tokens.

For local testing, `.env` and `.streamlit/secrets.toml` are gitignored, so either is a safe
place to stash your own `HF_TOKEN` without risking a commit.