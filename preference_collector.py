import streamlit as st
#import anthropic
from openai import OpenAI
import json
import csv
import os
from datetime import datetime, timezone

# ── Config ──────────────────────────────────────────────────────────────────
MODEL = "llama3.2:1b" #"claude-haiku-4-5-20251001"
DATA_FILE = "preference_data.jsonl"
CSV_FILE = "preference_data.csv"

st.set_page_config(page_title="Preference Collector", page_icon="⚖️", layout="wide")
st.title("⚖️ Response Preference Collector")
st.caption("Generate two responses, pick the better one — data saved for RLHF training.")

# ── Helpers ──────────────────────────────────────────────────────────────────
def generate_response(client: OpenAI, prompt: str) -> tuple[str, dict]:
    start = datetime.now(timezone.utc)
    msg = client.chat.completions.create(
        model=MODEL,
        temperature=1,
        messages=[{"role": "user", "content": prompt}],
    )
    end = datetime.now(timezone.utc)
    text = msg.choices[0].message.content
    meta = {
        "model": msg.model,
        "input_tokens": msg.usage.prompt_tokens,
        "output_tokens": msg.usage.completion_tokens,
        "latency_ms": int((end - start).total_seconds() * 1000),
        "timestamp": start.isoformat(),
    }
    return text, meta


def save_record(record: dict):
    """Append one JSONL row and keep the CSV in sync."""
    with open(DATA_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    write_header = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "prompt", "chosen", "rejected",
                        "model", "input_tokens", "output_tokens_chosen",
                        "output_tokens_rejected", "latency_ms_chosen", "latency_ms_rejected"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": record["timestamp"],
            "prompt": record["prompt"],
            "chosen": record["chosen"],
            "rejected": record["rejected"],
            "model": record["metadata"]["model"],
            "input_tokens": record["metadata"]["input_tokens"],
            "output_tokens_chosen": record["metadata"]["output_tokens_chosen"],
            "output_tokens_rejected": record["metadata"]["output_tokens_rejected"],
            "latency_ms_chosen": record["metadata"]["latency_ms_chosen"],
            "latency_ms_rejected": record["metadata"]["latency_ms_rejected"],
        })


def load_history() -> list[dict]:
    if not os.path.exists(DATA_FILE):
        return []
    records = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Session state init ────────────────────────────────────────────────────────
for key in ("resp_a", "resp_b", "meta_a", "meta_b", "current_prompt", "saved"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── API key ───────────────────────────────────────────────────────────────────
# api_key = st.sidebar.text_input("Anthropic API Key", type="password",
#                                  help="Never stored — only used for this session.")
# if not api_key:
#     st.info("Enter your Anthropic API key in the sidebar to get started.")
#     st.stop()

# client = anthropic.Anthropic(api_key=api_key)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# ── Prompt input & generation ────────────────────────────────────────────────
st.subheader("1 · Enter a prompt")
prompt = st.text_area("Prompt", height=100, placeholder="Ask anything…")

if st.button("🎲 Generate Two Responses", type="primary", disabled=not prompt.strip()):
    with st.spinner("Generating response A…"):
        ra, ma = generate_response(client, prompt)
    with st.spinner("Generating response B…"):
        rb, mb = generate_response(client, prompt)
    st.session_state.update(
        resp_a=ra, resp_b=rb, meta_a=ma, meta_b=mb,
        current_prompt=prompt, saved=False,
    )

# ── Display & preference selection ───────────────────────────────────────────
if st.session_state.resp_a:
    st.divider()
    st.subheader("2 · Compare responses")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Response A")
        st.text_area("", value=st.session_state.resp_a, height=300,
                     key="display_a", disabled=True)
        with st.expander("Metadata"):
            st.json(st.session_state.meta_a)

    with col_b:
        st.markdown("### Response B")
        st.text_area("", value=st.session_state.resp_b, height=300,
                     key="display_b", disabled=True)
        with st.expander("Metadata"):
            st.json(st.session_state.meta_b)

    st.divider()
    st.subheader("3 · Select your preference")
    preference = st.radio(
        "Which response is better?",
        ["Response A", "Response B", "Tie / Skip"],
        horizontal=True,
        key="preference_radio",
    )

    if st.button("✅ Save Preference", disabled=st.session_state.saved):
        if preference == "Tie / Skip":
            st.warning("Record skipped — no preference saved.")
        else:
            chosen  = st.session_state.resp_a if preference == "Response A" else st.session_state.resp_b
            rejected = st.session_state.resp_b if preference == "Response A" else st.session_state.resp_a
            meta_c = st.session_state.meta_a if preference == "Response A" else st.session_state.meta_b
            meta_r = st.session_state.meta_b if preference == "Response A" else st.session_state.meta_a

            record = {
                "timestamp": st.session_state.meta_a["timestamp"],
                "prompt": st.session_state.current_prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "model": st.session_state.meta_a["model"],
                    "input_tokens": st.session_state.meta_a["input_tokens"],
                    "output_tokens_chosen": meta_c["output_tokens"],
                    "output_tokens_rejected": meta_r["output_tokens"],
                    "latency_ms_chosen": meta_c["latency_ms"],
                    "latency_ms_rejected": meta_r["latency_ms"],
                },
            }
            save_record(record)
            st.session_state.saved = True
            st.success(f"Saved! **{preference}** marked as chosen.")

# ── Export & history ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📦 Saved Data")

history = load_history()
st.write(f"**{len(history)} record(s)** saved in this session / run.")

col1, col2 = st.columns(2)
with col1:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "rb") as f:
            st.download_button("⬇️ Download JSONL", f, file_name="preference_data.jsonl",
                               mime="application/jsonl")
with col2:
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "rb") as f:
            st.download_button("⬇️ Download CSV", f, file_name="preference_data.csv",
                               mime="text/csv")

if history:
    with st.expander("Preview last 5 records"):
        for rec in history[-5:][::-1]:
            st.markdown(f"**{rec['timestamp']}** — `{rec['prompt'][:80]}…`")
            st.caption(f"Chosen (first 120 chars): {rec['chosen'][:120]}")
            st.divider()