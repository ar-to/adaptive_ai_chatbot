from transformers import pipeline 
import streamlit as st
import random
import time

def simulate_typing(response):
    """Simulate typing effect for the assistant's response."""
    for chunk in response.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.markdown(full_response + "▌")
    message_placeholder.markdown(full_response)

# Initialize the sentiment analysis model (cached to prevent reloading every rerun)
@st.cache_resource
def load_sentiment_pipeline():
  # Using a robust, standard model for positive/negative/neutral tracking
  return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

analyzer = load_sentiment_pipeline()

st.title("🤖 Sentiment-Aware Chatbot")

st.caption("Note that this demo app isn't actually connected to any LLMs. Those are expensive ;)")

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

    # Run real-time sentiment analysis via Hugging Face
    analysis_result = analyzer(prompt)[0]
    user_sentiment = analysis_result["label"].lower() # returns 'positive', 'negative', or 'neutral'

    # Display the sentiment badge directly under the user's message
    st.caption(f" Detected Sentiment: {user_sentiment.upper()} (Confidence: {analysis_result['score']:.2f})")

    # Add user message to chat history with sentiment
    # Store the sentiment in the session state for potential future use
    st.session_state.messages.append({"role": "user", "content": prompt, "sentiment": user_sentiment})
    
    # 3. Generate adaptive bot response based on emotional context
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
        # assistant_response = random.choice(
        #     [
        #         "Hello there! How can I assist you today?",
        #         "Hi, human! Is there anything I can help you with?",
        #         "Do you need help?",
        #     ]
        # )

        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        # simulate_typing(assistant_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
