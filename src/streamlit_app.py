from transformers import pipeline
from huggingface_hub import InferenceClient
import streamlit as st
import os
import time

# HF Inference Providers' list of serverless-supported models rotates over time;
# if calls start 404ing, swap this for a currently supported chat-completion model.
DEFAULT_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Initialize the sentiment analysis model (cached to prevent reloading every rerun)
@st.cache_resource
def load_sentiment_pipeline():
  # Using a robust, standard model for positive/negative/neutral tracking
  return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Initialize the emotion classification model (cached to prevent reloading every rerun)
@st.cache_resource
def load_emotion_pipeline():
  # Fine-grained emotions (joy, sadness, anger, fear, surprise, disgust, neutral)
  return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

analyzer = load_sentiment_pipeline()
emotion_analyzer = load_emotion_pipeline()

def predict_sentiment(prompt):
    """Run sentiment inference, returning (label, confidence)."""
    result = analyzer(prompt)[0]
    return result["label"].lower(), result["score"]  # 'positive', 'negative', or 'neutral'

def predict_emotion(prompt):
    """Run emotion inference, returning (label, confidence)."""
    result = emotion_analyzer(prompt)[0]
    return result["label"].lower(), result["score"]  # e.g. 'joy', 'anger', 'sadness', 'fear'...

def get_llm_response(prompt, history, sentiment, emotion, token):
    """Call the Hugging Face Inference API using the caller's own token."""
    client = InferenceClient(model=DEFAULT_HF_MODEL, token=token)

    system_message = {
        "role": "system",
        "content": (
            "You are a warm, supportive chatbot. The user's latest message was "
            f"detected as having {sentiment} sentiment and {emotion} emotion, so "
            "respond with appropriate tone and empathy."
        ),
    }
    recent_history = [
        {"role": m["role"], "content": m["content"]} for m in history[-6:]
    ]

    response = client.chat_completion(
        messages=[system_message] + recent_history + [{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return response.choices[0].message.content

st.title("🤖 Sentiment-Aware Chatbot")

st.caption("Free by default with local sentiment analysis. Add a Hugging Face token below for real AI replies.")

# Resolve the active HF token: prefer a Space secret (env var) over a pasted one.
env_token = os.environ.get("HF_TOKEN")
if env_token:
    st.sidebar.success("AI replies enabled")
else:
    st.sidebar.text_input(
        "Hugging Face Token",
        type="password",
        key="user_hf_token",
        help="Get a free token at https://huggingface.co/settings/tokens",
    )
    st.sidebar.caption(
        "Paste a token to enable real AI replies for this session only (never stored). "
        "Or click 'Duplicate this Space', make it private, and add HF_TOKEN as a secret."
    )
active_token = env_token or st.session_state.get("user_hf_token")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run real-time sentiment and emotion analysis via Hugging Face
    user_sentiment, sentiment_score = predict_sentiment(prompt)
    user_emotion, emotion_score = predict_emotion(prompt)

    # Display the sentiment and emotion badges directly under the user's message
    st.caption(
        f" Detected Sentiment: {user_sentiment.upper()} (Confidence: {sentiment_score:.2f}) · "
        f"Emotion: {user_emotion.upper()} (Confidence: {emotion_score:.2f})"
    )

    # Add user message to chat history with sentiment and emotion
    # Store both in the session state for potential future use
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "sentiment": user_sentiment, "emotion": user_emotion}
    )

    # Generate adaptive bot response: real LLM reply if a token is available, canned fallback otherwise
    if active_token:
        try:
            assistant_response = get_llm_response(
                prompt, st.session_state.messages, user_sentiment, user_emotion, active_token
            )
        except Exception:
            st.error("Couldn't reach the model — check that your token is valid, or try again in a moment.")
            if "positive" in user_sentiment:
                assistant_response = "That sounds amazing! I'm incredibly happy to hear that. 🎉"
            elif "negative" in user_sentiment:
                assistant_response = "I am so sorry to hear that. I'm here if you want to vent or talk through it. ❤️"
            else:
                assistant_response = "Thanks for sharing that with me. Tell me more! 💬"
    else:
        if "positive" in user_sentiment:
            assistant_response = "That sounds amazing! I'm incredibly happy to hear that. 🎉"
        elif "negative" in user_sentiment:
            assistant_response = "I am so sorry to hear that. I'm here if you want to vent or talk through it. ❤️"
        else:
            assistant_response = "Thanks for sharing that with me. Tell me more! 💬"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
