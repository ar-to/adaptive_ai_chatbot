# Hugging Face Token Handling

The app needs an HF token to call the Inference Providers API for real LLM replies (`get_llm_response` in [streamlit_app.py](../src/streamlit_app.py)). Without one, the app still runs, just falling back to canned responses based on detected sentiment.

There are two ways a token can reach the app, resolved in [streamlit_app.py:80-97](../src/streamlit_app.py#L80-L97):

## 1. Space secret (`HF_TOKEN` env var)

- Set by whoever deploys the Space, via the Space's Settings → Repository secrets.
- Read once at startup with `os.environ.get("HF_TOKEN")`.
- If present, it's treated as a **shared token for every visitor** — the sidebar skips the token input entirely and just shows `AI replies enabled`.
- Use this for a private/duplicated Space where the deployer is fine paying for everyone's usage.

## 2. Pasted personal token (per session)

- Shown only when `HF_TOKEN` is not set.
- Sidebar renders a password-style `st.text_input` (key `user_hf_token`); Streamlit stores whatever the visitor types into `st.session_state["user_hf_token"]`.
- Scoped to that visitor's session only — never written to disk or shared across users.
- Sidebar shows `AI replies enabled` once a token has been entered.

## Resolution

```python
active_token = env_token or st.session_state.get("user_hf_token")
```

The env var always wins if set. `active_token` is what's passed to `get_llm_response(...)`; if it's falsy (neither source set), the app skips the LLM call and uses the canned sentiment-based response instead.

## Known gotcha: HF Inference Providers billing

Even with a valid token, calls can fail with `402 Payment Required — depleted your monthly included credits`. This is an HF account/billing limit, not a bug in this app:

- HF Pro includes a small monthly credit allowance for Inference Providers (a shared pool across all providers, not unlimited).
- Once exhausted, further calls 402 until the allowance resets or pay-as-you-go billing is configured on the HF account.
- Testing a provider's endpoint directly (e.g. `router.huggingface.co/novita/...`) is a different code path than the HF unified router the app uses, and can fail for unrelated reasons (e.g. provider-specific model naming) — it doesn't confirm or rule out the credits issue.

### Gotcha in practice: hitting a provider's endpoint directly

```bash
curl https://router.huggingface.co/novita/v3/openai/chat/completions \
  -H "Authorization: Bearer <hf_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
# {"error":"Model not supported by provider novita"}
```

This isn't the same request `InferenceClient.chat_completion(...)` makes. `router.huggingface.co/novita/...` is Novita's **own** OpenAI-compatible endpoint, passed through HF — it expects Novita's native model naming (e.g. lowercase `meta-llama/llama-3.1-8b-instruct`), not HF's canonical model ID. `InferenceClient` instead calls HF's **unified** router, which looks up the model's `inferenceProviderMapping` and translates the ID for whichever provider it picks automatically. So a 402 from the app and a "model not supported" error from a raw provider-endpoint curl are unrelated failures from two different code paths — the curl above doesn't validate or rule out the app's billing issue.

## Transformers API vs. Inference Providers API

This app uses both, and they're unrelated in every way that matters (cost, network, failure modes):

| | Sentiment/emotion (`load_sentiment_pipeline`, `load_emotion_pipeline`) | Chat replies (`get_llm_response`) |
|---|---|---|
| Library | [`transformers.pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines) | [`huggingface_hub.InferenceClient`](https://huggingface.co/docs/huggingface_hub/guides/inference) |
| Where it runs | Locally, in-process — model weights are downloaded once and executed on whatever machine hosts the Streamlit app | Remote — HTTP call to `router.huggingface.co`, routed to a hosted [Inference Provider](https://huggingface.co/docs/inference-providers/index) (e.g. [Novita](https://huggingface.co/docs/inference-providers/providers/novita)) |
| Needs a token? | No | Yes |
| Billed? | No (just local compute/memory) | Yes — subject to HF's monthly included credits / PAYG |
| Can 402? | No | Yes |

See also: [Inference Providers pricing docs](https://huggingface.co/docs/inference-providers/pricing).
