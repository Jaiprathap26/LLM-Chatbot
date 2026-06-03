import streamlit as st
import uuid
import os
from dotenv import load_dotenv
import logging

from memory.long_term_memory import LongTermMemory
from memory.buffer_memory import BufferMemory
from guardrails.model_armor import ModelArmorGuardrails
from utils.gemini_client import GeminiClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

BUFFER_SIZE = int(os.getenv("BUFFER_SIZE", "10"))
RECALL_K = int(os.getenv("RECALL_K", "3"))

st.set_page_config(page_title="LLM-Chatbot", page_icon="🤖")


@st.cache_resource
def init_services():
    ltm = LongTermMemory()
    buffer_mem = BufferMemory(ltm)
    guardrails = ModelArmorGuardrails()
    gemini = GeminiClient()
    return ltm, buffer_mem, guardrails, gemini


try:
    ltm, buffer_mem, guardrails, gemini = init_services()
except Exception as e:
    st.error(f"Failed to initialize services: {e}")
    st.stop()

st.title("🤖 Enterprise LLM Chatbot")

# Session management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

st.sidebar.header("Session Settings")
user_session_id = st.sidebar.text_input(
    "Session ID", value=st.session_state.session_id)
if user_session_id != st.session_state.session_id:
    st.session_state.session_id = user_session_id
    st.rerun()

st.sidebar.text(f"Active Session:\n{st.session_state.session_id}")

# Token tracking
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0


# Cost tracking
INPUT_COST_PER_M = 0.075
OUTPUT_COST_PER_M = 0.30

if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0

if "latest_message_cost" not in st.session_state:
    st.session_state.latest_message_cost = 0.0

st.sidebar.header("Usage Stats")
st.sidebar.text(f"Input Tokens: {st.session_state.total_input_tokens}")
st.sidebar.text(f"Output Tokens: {st.session_state.total_output_tokens}")
st.sidebar.text(f"Session Cost: ${st.session_state.session_cost:.6f}")
st.sidebar.text(f"Latest Cost: ${st.session_state.latest_message_cost:.6f}")


# Display buffer chat history
buffer_history = buffer_mem.get_recent_messages(
    st.session_state.session_id, limit=BUFFER_SIZE)

for msg in buffer_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("What is up?"):

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. Guardrails check
    with st.spinner("Checking safety..."):
        safety_check = guardrails.sanitize_input(prompt)

    if not safety_check["is_safe"]:
        with st.chat_message("assistant"):
            st.error(safety_check["sanitized_text"])
            ltm.save_message(
                st.session_state.session_id,
                "user",
                prompt,
                embedding=None)
            ltm.save_message(
                st.session_state.session_id,
                "assistant",
                safety_check["sanitized_text"],
                embedding=None)
    else:
        safe_prompt = safety_check["sanitized_text"]

        # 2. Get Embedding for the user prompt
        with st.spinner("Generating embedding..."):
            embedding = gemini.generate_embedding(safe_prompt)

        # 3. Retrieve relevant long-term memories
        with st.spinner("Retrieving memories..."):
            historical_memories = []
            if embedding:
                historical_memories = ltm.get_relevant_memories(
                    st.session_state.session_id,
                    embedding,
                    limit=RECALL_K,
                    exclude_last_n=BUFFER_SIZE
                )

        # Add new user message to the temporary context we will send to Gemini
        # We append to buffer_history for the Gemini context
        context_messages = buffer_history.copy()
        context_messages.append({"role": "user", "content": safe_prompt})

        # 4. Generate Response
        with st.spinner("Thinking..."):
            response_data = gemini.generate_chat_response(
                context_messages, historical_memories)

        # 5. Display Response
        with st.chat_message("assistant"):
            st.markdown(response_data["response_text"])

        # Update stats
        st.session_state.total_input_tokens += response_data["input_tokens"]
        st.session_state.total_output_tokens += response_data["output_tokens"]

        # Calculate cost
        in_cost = (response_data["input_tokens"] /
                   1_000_000) * INPUT_COST_PER_M
        out_cost = (response_data["output_tokens"] /
                    1_000_000) * OUTPUT_COST_PER_M
        msg_cost = in_cost + out_cost

        st.session_state.latest_message_cost = msg_cost
        st.session_state.session_cost += msg_cost

        # 6. Save to DB
        # Save user message
        ltm.save_message(
            st.session_state.session_id,
            "user",
            prompt,
            embedding=embedding)

        # Save assistant message (also get its embedding for future recall)
        # Note: Depending on cost/time, you might skip embedding the assistant response,
        # but embedding it allows semantic search over what the assistant said
        # too.
        assistant_embedding = gemini.generate_embedding(
            response_data["response_text"])
        ltm.save_message(
            st.session_state.session_id,
            "assistant",
            response_data["response_text"],
            embedding=assistant_embedding)

        st.rerun()
