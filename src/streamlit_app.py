from transformers import pipeline
from huggingface_hub import InferenceClient
import streamlit as st
import traceback
import colorsys
import os
import random
import time

# MODELS
# HF Inference Providers' list of serverless-supported models rotates over time;
# if calls start 404ing, swap this for a currently supported chat-completion model.
DEFAULT_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

# Pinned local transformers models, referenced by both the app and its tests
# (see adaptive_ai_chatbot/docs/live-inference-testing.md).
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# MODELS END
# taken from Gemini (Google AI Mode) gen

# COLOR & TRAITS
# 1. Base Symbolic Anchors (Emotion -> Color Space Angles in Degrees)
EMOTION_HUES = {
    "joy": 45,       # Warm Amber/Yellow
    "sadness": 220,  # Deep Blue (Psychology consensus)
    "anger": 0,      # Urgent Red
    "fear": 280,     # Muted Purple
    "surprise": 180, # Bright Teal
    "disgust": 100,  # Olive Green
    "neutral": 0,    # Desaturated Slate Gray
}

# 2. Domain Knowledge Matrices
DOMAIN_MODIFIERS = {
    "medical": {
        "saturation_cap": 0.40,  # Muted, calming tones to lower cognitive load
        "value_floor": 0.85,     # Light, clean backgrounds
        "hue_shift": {"sadness": 200} # Shift sadness to soft soothing teals
    },
    "marketing": {
        "saturation_cap": 0.95,  # High urgency, vibrant colors
        "value_floor": 0.60,     # Striking, high-contrast layouts
        "hue_shift": {"anger": 10} # Pure fiery orange-red
    },
    "general": {
        "saturation_cap": 0.70,
        "value_floor": 0.80,
        "hue_shift": {}
    }
}


def generate_adaptive_palette(user_sentiment, emotion_breakdown, active_domain="general", trait_sliders=None):
    """
    Symbolic core evaluating neural vector outputs against domain configurations
    and runtime trait metrics to resolve a deterministic hex theme.
    """
    if trait_sliders is None:
        trait_sliders = {"psychological": 1.0, "philosophical": 0.0, "support_ambiguity": False}
        
    config = DOMAIN_MODIFIERS.get(active_domain, DOMAIN_MODIFIERS["general"])
    
    # Calculate a weighted Hue angle across the entire emotion probability distribution
    total_score = 0.0
    weighted_hue_x = 0.0
    weighted_hue_y = 0.0
    
    for item in emotion_breakdown:
        emotion = item["label"].lower()
        score = item["score"]
        
        # Pull mapped base Hue angle
        hue_angle = EMOTION_HUES.get(emotion, 0)
        # Apply domain shifts dynamically 
        hue_angle = config["hue_shift"].get(emotion, hue_angle)

        # Convert angle to vector coordinates to resolve many-to-many blending accurately
        import math
        rad = math.radians(hue_angle)
        weighted_hue_x += score * math.cos(rad)
        weighted_hue_y += score * math.sin(rad)
        total_score += score

    # Determine mixed hue angle from blended vectors
    if total_score > 0:
        avg_rad = math.atan2(weighted_hue_y, weighted_hue_x)
        final_hue = (math.degrees(avg_rad) % 360) / 360.0
    else:
        final_hue = 0.0

    # Handle Ambiguity
    if trait_sliders.get("support_ambiguity", False):
      final_hue_angle_degrees = (final_hue * 360) % 360 # convert back to degrees for easier reasoning
      # Skew the final calculated Hue angle based on the macro sentiment anchor
      if user_sentiment == "positive" and final_hue_angle_degrees > 180:
          # Force alignment towards the warmer, uplifting spectrum (e.g., yellows/greens)
          final_hue = (final_hue + 0.15) % 1.0 
      elif user_sentiment == "negative" and final_hue_angle_degrees < 90:
          # Shift away from bright energetic zones into protective deep blues/muted plums
          final_hue = (final_hue + 0.40) % 1.0
    # Handle Ambiguity END
    

    # Extract dominant emotion score to drive saturation depth
    dominant_emotion = emotion_breakdown[0]["label"].lower()
    dominant_score = emotion_breakdown[0]["score"]
    
    # Modulate color profile traits using runtime interface sliders
    # Higher philosophical trait = cooler, deeper saturation tones
    sat_multiplier = 1.0 - (trait_sliders.get("philosophical", 0.0) * 0.3)
    
    # Enforce safe domain limits
    final_sat = min(dominant_score * sat_multiplier, config["saturation_cap"])
    if dominant_emotion == "neutral":
        final_sat = 0.05 # Keep slate tones completely clean
        
    final_val = max(1.0 - (dominant_score * 0.2), config["value_floor"])
    
    # Convert HSV back to Hexadecimal
    r, g, b = colorsys.hsv_to_rgb(final_hue, final_sat, final_val)
    hex_bg = f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
    
    # Derive corresponding semantic contrast rules (Text and Borders)
    hex_text = "#111827" if final_val > 0.5 else "#F9FAFB"
    hex_accent = f"#{int(r*255*0.7):02X}{int(g*255*0.7):02X}{int(b*255*0.7):02X}"

    # TODO: need to map HEX to a Color Label e.g. "Warm Amber", "Deep Blue", "Muted Teal" for user-facing feedback. Maybe via local model?
    
    return {
        "background": hex_bg,
        "text": hex_text,
        "accent": hex_accent,
        "input_bg": "#FFFFFF" if final_val > 0.5 else "#1F2937",
        "input_border": hex_accent
    }

def hue_to_hex(hue_degrees: float, sat: float = 0.65, val: float = 0.9) -> str:
    """Render a bright, legible swatch color for a raw EMOTION_HUES angle."""
    r, g, b = colorsys.hsv_to_rgb((hue_degrees % 360) / 360.0, sat, val)
    return f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"

def build_css(theme: dict) -> str:
    return f"""
    <style>
        .stApp {{
            background-color: {theme["background"]} !important;
            transition: background-color 0.5s ease;
        }}
        [data-testid="stHeader"] {{
            background-color: transparent !important;
        }}
        h1, h2, h3, p, label {{
            color: {theme["text"]} !important;
        }}
        [data-testid="stCaptionContainer"] {{
            color: {theme["text"]} !important;
            opacity: 0.75;
        }}
        [data-testid="stTextInputRootElement"] {{
            background-color: {theme["input_bg"]} !important;
            border-color: {theme["input_border"]} !important;
        }}
        [data-testid="stTextInputRootElement"] input {{
            color: {theme["text"]} !important;
            background-color: transparent !important;
        }}
        [data-testid="stTextInputRootElement"] input::placeholder {{
            color: {theme["text"]} !important;
            opacity: 0.5;
        }}
        [data-testid="stButton"] button {{
            background-color: {theme["accent"]} !important;
            color: #FFFFFF !important;
            border: none !important;
        }}
    </style>
    """

# COLOR & TRAITS END 

# Shown in the spinner while a message is being analyzed and answered.
# Tweak this list freely, or swap get_loading_message() below for an AI call
# (e.g. request a one-line joke) to make the wait more creative.
LOADING_MESSAGES = [
    "Reading the room... 🧠",
    "Weighing your words... ⚖️",
    "Consulting the emotion oracle... 🔮",
    "Thinking of something kind to say... 💬",
]

def get_loading_message():
    """Text shown in the spinner while a message is analyzed and answered."""
    return random.choice(LOADING_MESSAGES)

# Initialize the sentiment analysis model (cached to prevent reloading every rerun)
@st.cache_resource
def load_sentiment_pipeline():
  # Using a robust, standard model for positive/negative/neutral tracking
  return pipeline("text-classification", model=SENTIMENT_MODEL)

# Initialize the emotion classification model (cached to prevent reloading every rerun)
@st.cache_resource
def load_emotion_pipeline():
  # Fine-grained emotions (joy, sadness, anger, fear, surprise, disgust, neutral)
  return pipeline("text-classification", model=EMOTION_MODEL)

analyzer = load_sentiment_pipeline()
emotion_analyzer = load_emotion_pipeline()

def predict_sentiment(prompt):
    """Run sentiment inference once, returning (label, confidence, full breakdown)."""
    breakdown = sorted(analyzer(prompt, top_k=None), key=lambda r: r["score"], reverse=True)
    top = breakdown[0]
    # e.g. Sentiment breakdown: [{'label': 'neutral', 'score': 0.5090614557266235}, {'label': 'positive', 'score': 0.40823495388031006}, {'label': 'negative', 'score': 0.08270362764596939}]
    # print(f"Sentiment breakdown: {breakdown}")
    return top["label"].lower(), top["score"], breakdown  # label is 'positive', 'negative', or 'neutral'

def predict_emotion(prompt):
    """Run emotion inference once, returning (label, confidence, full breakdown)."""
    breakdown = sorted(emotion_analyzer(prompt, top_k=None), key=lambda r: r["score"], reverse=True)
    top = breakdown[0]
    return top["label"].lower(), top["score"], breakdown  # e.g. 'joy', 'anger', 'sadness', 'fear'...

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

# UI/interactive code only runs under `streamlit run` (which executes this file
# with __name__ == "__main__"), not on a plain `import streamlit_app` — this lets
# tests import the module for its pipelines/functions without touching Streamlit's
# widget APIs. See adaptive_ai_chatbot/docs/live-inference-testing.md.
if __name__ == "__main__":
    st.title("🤖 Sentiment-Aware Chatbot")

    st.caption("Free by default with local sentiment analysis. Add a Hugging Face token below for real AI replies using an LLM.")

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
        # intentional to keep UI about open but still show user a success
        if st.session_state.get("user_hf_token"):
            st.sidebar.success("AI replies enabled")
    active_token = env_token or st.session_state.get("user_hf_token")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # DOMAINS & SLIDERS
    # Sidebar Controller for Trait Profiles & Domain Settings
    st.sidebar.header("🎛️ Neurosymbolic Adaptivity Controls")
    selected_domain = st.sidebar.selectbox("Active Interface Domain", ["general", "medical", "marketing"])
    
    slider_psych = st.sidebar.slider("Psychological Focus Weight", 0.0, 1.0, 0.7)
    slider_philo = st.sidebar.slider(
        "Philosophical Reflection Weight", 0.0, 1.0, 0.2,
        help="Higher philosophical trait = cooler, deeper saturation tones"
    )
    support_ambiguity_toggle = st.sidebar.checkbox(
        "Support Ambiguity Handling", value=False,
        help="Enable handling of ambiguous emotions. Skew mixed-emotion hues towards the sentiment's expected color spectrum. e.g. positive -> warm yellows/greens, negative -> deep blues/muted plums"
    )

    runtime_traits = {"psychological": slider_psych, "philosophical": slider_philo, "support_ambiguity": support_ambiguity_toggle}
    # DOMAINS & SLIDERS END

    # Accept user input
    if prompt := st.chat_input("What is up?"):

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Block on a spinner while analysis + response generation run, so the user
        # can't fire off a second prompt mid-flight and race the chat history.
        with st.spinner(get_loading_message()):

            # add a small delay to simulate processing time and improve UX
            time.sleep(0.5)

            # Run real-time sentiment and emotion analysis via Hugging Face (one pass per model)
            user_sentiment, sentiment_score, sentiment_breakdown = predict_sentiment(prompt)
            user_emotion, emotion_score, emotion_breakdown = predict_emotion(prompt)
            # Store the last detected emotion in session state for the sidebar legend
            st.session_state.last_emotion = user_emotion.lower()

            # Generate adaptive bot response: real LLM reply if a token is available, canned fallback otherwise
            if active_token:
                try:
                    assistant_response = get_llm_response(
                        prompt, st.session_state.messages, user_sentiment, user_emotion, active_token
                    )
                except Exception:
                    # print actual error to console for debugging, but show a user-friendly message in the UI
                    traceback.print_exc()
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

        # Display the sentiment and emotion badges directly under the user's message
        st.caption(
            f" Detected Sentiment: {user_sentiment.upper()} (Confidence: {sentiment_score:.2f}) · "
            f"Emotion: {user_emotion.upper()} (Confidence: {emotion_score:.2f})"
        )

        # Full label breakdown in a collapsible dropdown, below the caption
        with st.expander("🔍 Full analysis breakdown"):
            st.markdown("**Sentiment scores**")
            st.dataframe(
                [{"label": r["label"], "confidence": f"{r['score']:.2%}"} for r in sentiment_breakdown],
                hide_index=True,
                use_container_width=True,
            )
            st.markdown("**Emotion scores**")
            st.dataframe(
                [{"label": r["label"], "confidence": f"{r['score']:.2%}"} for r in emotion_breakdown],
                hide_index=True,
                use_container_width=True,
            )

        # Add user message to chat history with sentiment and emotion
        # Store both in the session state for potential future use
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "sentiment": user_sentiment, "emotion": user_emotion}
        )

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

        EMOJI = {"positive": "😊", "negative": "😔", "neutral": "😐"}
        if prompt.strip():
            st.subheader(f"{EMOJI[user_sentiment]} Detected sentiment: {user_sentiment.title()}")
            # st.caption(f"{pos_score} positive cue(s), {neg_score} negative cue(s) matched")
        else:
            st.caption("Waiting for input to analyze sentiment…")


        # FAST RUNTIME ADAPTATION LAYER
        # Pass both sentiment and emotion to the neurosymbolic layout matrix
        theme = generate_adaptive_palette(
            user_sentiment=user_sentiment,            # Controls macro contrast/bounds
            emotion_breakdown=emotion_breakdown,      # Controls micro hue calculations
            active_domain=selected_domain,            # Sets domain constraints
            trait_sliders=runtime_traits            # Modulates runtime scaling weights
        )

        # print(f"DEBUG: Generated theme: {theme} for sentiment={user_sentiment}, emotion={user_emotion}, domain={selected_domain}, traits={runtime_traits}")

        # Apply calculated style modifications immediately
        st.markdown(build_css(theme), unsafe_allow_html=True)
    st.caption("POC note: sentiment detection uses simple keyword matching — swap in a fine-tuned model for production.")

    # EMOTION HUE LEGEND
    # Live reference of each emotion's base hue, with the most recently detected
    # emotion (if any) highlighted so the mapping stays visible as you chat.
    # Rendered after the chat_input block so it reflects this run's freshly
    # detected emotion rather than the previous run's stale session state.
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Emotion Hue Legend")
    current_emotion = st.session_state.get("last_emotion")
    for emotion, hue in EMOTION_HUES.items():
        swatch = hue_to_hex(hue)
        is_current = emotion == current_emotion
        row_style = f"font-weight:700;background:{swatch}22;border-radius:4px;padding:2px 4px;" if is_current else "padding:2px 4px;"
        st.sidebar.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;{row_style}'>"
            f"<span style='width:14px;height:14px;border-radius:50%;background:{swatch};"
            f"border:1px solid rgba(0,0,0,0.25);flex-shrink:0;'></span>"
            f"<span>{emotion.title()} ({hue}°){' ← detected' if is_current else ''}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # EMOTION HUE LEGEND END