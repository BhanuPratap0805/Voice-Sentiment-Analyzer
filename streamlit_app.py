import streamlit as st
import whisper
from transformers import pipeline
import os
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Voice Sentiment Analyzer",
    page_icon="🎙️",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Cached Model Loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_whisper_model():
    """Load Whisper model once and cache it."""
    print("[*] Loading Whisper speech-recognition model (base)...")
    return whisper.load_model("base")

@st.cache_resource
def load_emotion_model():
    """Load Emotion model once and cache it."""
    print("[*] Loading emotion classifier...")
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
    )

# Load models
with st.spinner("Loading models (this may take a moment on first run)..."):
    asr_model = load_whisper_model()
    emotion_clf = load_emotion_model()

# ---------------------------------------------------------------------------
# Emotion Config
# ---------------------------------------------------------------------------
EMOTION_COLORS = {
    "joy": "#FFD700",
    "sadness": "#4169E1",
    "anger": "#DC143C",
    "fear": "#8B008B",
    "neutral": "#808080",
    "surprise": "#FF8C00",
    "disgust": "#228B22",
    "excitement": "#FF1493",
}

EMOTION_EMOJI = {
    "joy": "😄", "sadness": "😢", "anger": "😡",
    "fear": "😨", "neutral": "😐", "surprise": "😲",
    "disgust": "🤢", "excitement": "🤩",
}

# Neutral dampening factor — the text model is heavily biased toward
# "neutral" for short sentences.  Multiplying the raw neutral score by
# this value lets genuine emotions surface when the model is uncertain.
NEUTRAL_DAMPEN = 0.6

# Minimum window duration (seconds) for merging short segments.
# Whisper segments can be 1‑3 words; merging them into ≥ this length
# gives the emotion classifier enough context to work properly.
MERGE_WINDOW_SEC = 12.0

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _fmt(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _check_excitement(scores: dict[str, float]) -> tuple[bool, float]:
    """Detect excitement from high joy + surprise combination."""
    joy = scores.get("joy", 0)
    surprise = scores.get("surprise", 0)
    if (joy >= 0.20 and surprise >= 0.10) or (surprise >= 0.20 and joy >= 0.10):
        return True, round(joy * 0.6 + surprise * 0.4, 3)
    return False, 0.0


def merge_segments(raw_segments: list[dict], window_sec: float = MERGE_WINDOW_SEC) -> list[dict]:
    """Merge short Whisper segments into longer windows.

    Short segments (~1‑3 words) almost always classify as 'neutral'
    because the emotion model has too little context.  This function
    groups consecutive segments so that each merged window is at least
    *window_sec* seconds long, giving the classifier enough text to
    detect real emotions.
    """
    if not raw_segments:
        return []

    merged = []
    buf_texts: list[str] = []
    buf_start: float = raw_segments[0]["start"]
    buf_end: float = raw_segments[0]["end"]

    for seg in raw_segments:
        text = seg["text"].strip()
        if not text:
            continue

        if not buf_texts:
            buf_start = seg["start"]
            buf_end = seg["end"]
            buf_texts.append(text)
            continue

        current_len = buf_end - buf_start
        if current_len >= window_sec:
            # Flush current buffer
            merged.append({
                "start": buf_start,
                "end": buf_end,
                "text": " ".join(buf_texts),
            })
            buf_texts = [text]
            buf_start = seg["start"]
            buf_end = seg["end"]
        else:
            buf_texts.append(text)
            buf_end = seg["end"]

    # Flush remaining
    if buf_texts:
        merged.append({
            "start": buf_start,
            "end": buf_end,
            "text": " ".join(buf_texts),
        })

    return merged


def classify_emotion(text: str, emotion_clf) -> dict:
    """Run emotion classification with neutral dampening.

    Returns a dict with keys: emotion, confidence, scores.
    """
    emotions = emotion_clf(text)[0]
    score_dict = {e["label"]: e["score"] for e in emotions}

    # --- Dampen neutral bias ---
    if "neutral" in score_dict:
        score_dict["neutral"] *= NEUTRAL_DAMPEN

    # Re-normalise so scores still sum to ~1
    total = sum(score_dict.values())
    if total > 0:
        score_dict = {k: v / total for k, v in score_dict.items()}

    # --- Check excitement (high joy + surprise) ---
    is_excited, exc_conf = _check_excitement(score_dict)
    if is_excited:
        emo_label = "excitement"
        confidence = exc_conf
    else:
        dominant = max(score_dict, key=score_dict.get)
        emo_label = dominant
        confidence = score_dict[dominant]

    return {
        "emotion": emo_label,
        "confidence": round(confidence, 3),
        "scores": {k: round(v, 4) for k, v in score_dict.items()},
    }


def detect_change_points(segments: list[dict]) -> list[dict]:
    """Find timestamps where the dominant emotion changes."""
    changes = []
    for i in range(1, len(segments)):
        if segments[i]["emotion"] != segments[i - 1]["emotion"]:
            changes.append({
                "time": segments[i]["start"],
                "time_fmt": _fmt(segments[i]["start"]),
                "from": segments[i - 1]["emotion"],
                "to": segments[i]["emotion"],
            })
    return changes


# ---------------------------------------------------------------------------
# Main App UI
# ---------------------------------------------------------------------------
st.title("🎙️ Voice Sentiment Analyzer")
st.markdown("""
Upload an audio file to transcribe speech and analyze emotions over time.
Supported formats: **WAV, MP3, M4A, OGG**
""")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    if st.button("🔍 Analyze Audio", type="primary"):
        try:
            with st.spinner("Transcribing and Analyzing..."):
                # Save to temp file
                suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    # --- Transcribe ---
                    result = asr_model.transcribe(tmp_path)
                    raw_segments = result.get("segments", [])
                    full_text = result.get("text", "")

                    if not raw_segments:
                        st.error("No speech detected. Try a clearer recording.")
                        st.stop()

                    duration = raw_segments[-1]["end"]

                    # --- Merge segments for better emotion detection ---
                    merged = merge_segments(raw_segments, window_sec=MERGE_WINDOW_SEC)

                    # --- Emotion analysis on merged windows ---
                    segments: list[dict] = []
                    emo_dur: dict[str, float] = {}

                    for win in merged:
                        emo = classify_emotion(win["text"], emotion_clf)
                        seg_dur = win["end"] - win["start"]
                        emo_dur[emo["emotion"]] = emo_dur.get(emo["emotion"], 0) + seg_dur

                        segments.append({
                            "start": win["start"],
                            "end": win["end"],
                            "text": win["text"],
                            **emo,
                        })

                    # --- Also run fine‑grained (per original Whisper segment) ---
                    fine_segments: list[dict] = []
                    for seg in raw_segments:
                        text = seg["text"].strip()
                        if not text:
                            continue
                        # For fine‑grained, still use context: prepend / append
                        # neighbours so the classifier sees more text.
                        idx = raw_segments.index(seg)
                        context_texts = []
                        for j in range(max(0, idx - 2), min(len(raw_segments), idx + 3)):
                            t = raw_segments[j]["text"].strip()
                            if t:
                                context_texts.append(t)
                        context = " ".join(context_texts)

                        emo = classify_emotion(context, emotion_clf)
                        fine_segments.append({
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": text,
                            **emo,
                        })

                    # --- Detect change points ---
                    changes = detect_change_points(segments)

                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            # ================================================================
            # DISPLAY RESULTS
            # ================================================================

            st.success("✅ Analysis complete!")
            st.divider()

            # ---- 1. Overview Metrics ----
            st.subheader("📊 Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Duration", f"{duration:.1f}s")
            c2.metric("Words", len(full_text.split()))
            c3.metric("Segments", len(segments))
            if emo_dur:
                overall_dominant = max(emo_dur, key=emo_dur.get)
                emoji = EMOTION_EMOJI.get(overall_dominant, "😶")
                c4.metric("Dominant Emotion", f"{emoji} {overall_dominant.capitalize()}")

            st.divider()

            # ---- 2. Emotion Timeline Chart ----
            st.subheader("🕐 Emotion Timeline")
            st.caption("Shows how emotions change throughout the audio — each point is a segment midpoint.")

            # Build DataFrame for timeline
            timeline_rows = []
            for s in segments:
                mid = (s["start"] + s["end"]) / 2
                timeline_rows.append({
                    "Time (s)": round(mid, 1),
                    "Timestamp": _fmt(mid),
                    "Emotion": s["emotion"].capitalize(),
                    "Confidence": s["confidence"],
                    "Text": s["text"][:80] + ("…" if len(s["text"]) > 80 else ""),
                })
            timeline_df = pd.DataFrame(timeline_rows)

            # Assign a numeric Y value per emotion for the timeline
            unique_emotions = sorted(timeline_df["Emotion"].unique())
            emo_y_map = {e: i for i, e in enumerate(unique_emotions)}
            timeline_df["Emotion_Y"] = timeline_df["Emotion"].map(emo_y_map)

            fig_timeline = go.Figure()

            # Line connecting points
            fig_timeline.add_trace(go.Scatter(
                x=timeline_df["Time (s)"],
                y=timeline_df["Emotion_Y"],
                mode="lines",
                line=dict(color="rgba(150,150,150,0.4)", width=2, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ))

            # Colored scatter points
            for emo_name in unique_emotions:
                mask = timeline_df["Emotion"] == emo_name
                subset = timeline_df[mask]
                color = EMOTION_COLORS.get(emo_name.lower(), "#888888")
                fig_timeline.add_trace(go.Scatter(
                    x=subset["Time (s)"],
                    y=subset["Emotion_Y"],
                    mode="markers",
                    marker=dict(size=14, color=color, line=dict(width=1, color="white")),
                    name=emo_name,
                    text=subset["Text"],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Emotion: %{customdata[1]}<br>"
                        "Confidence: %{customdata[2]:.0%}<br>"
                        "Text: %{text}<extra></extra>"
                    ),
                    customdata=list(zip(subset["Timestamp"], subset["Emotion"], subset["Confidence"])),
                ))

            fig_timeline.update_layout(
                yaxis=dict(
                    tickvals=list(emo_y_map.values()),
                    ticktext=list(emo_y_map.keys()),
                    title="",
                ),
                xaxis=dict(title="Time (seconds)"),
                height=350,
                margin=dict(l=100, r=20, t=30, b=50),
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

            # ---- 3. Emotion Change Points ----
            if changes:
                st.subheader("⚡ Emotion Change Points")
                st.caption("Exact moments where the detected emotion shifts.")
                change_df = pd.DataFrame([
                    {
                        "⏱️ Timestamp": c["time_fmt"],
                        "Time (s)": c["time"],
                        "From": f"{EMOTION_EMOJI.get(c['from'], '😶')} {c['from'].capitalize()}",
                        "To": f"{EMOTION_EMOJI.get(c['to'], '😶')} {c['to'].capitalize()}",
                    }
                    for c in changes
                ])
                st.dataframe(change_df, use_container_width=True, hide_index=True)
            else:
                st.info("No emotion changes detected — the entire recording appears to have a consistent emotion.")

            st.divider()

            # ---- 4. Emotion Distribution ----
            col_pie, col_bar = st.columns(2)

            with col_pie:
                st.subheader("🥧 Emotion Distribution")
                if emo_dur:
                    pie_data = [{"Emotion": k.capitalize(), "Duration (s)": round(v, 1)} for k, v in emo_dur.items()]
                    fig_pie = px.pie(
                        pie_data, values="Duration (s)", names="Emotion",
                        color="Emotion",
                        color_discrete_map={k.capitalize(): v for k, v in EMOTION_COLORS.items()},
                        hole=0.45,
                    )
                    fig_pie.update_layout(height=350, margin=dict(t=20, b=20))
                    st.plotly_chart(fig_pie, use_container_width=True)

            with col_bar:
                st.subheader("📊 Emotion Scores per Segment")
                # Stacked bar chart showing all emotion scores per segment
                bar_rows = []
                for i, s in enumerate(segments):
                    for emo_key, emo_val in s["scores"].items():
                        bar_rows.append({
                            "Segment": f"{_fmt(s['start'])}–{_fmt(s['end'])}",
                            "Emotion": emo_key.capitalize(),
                            "Score": emo_val,
                        })
                bar_df = pd.DataFrame(bar_rows)
                fig_bar = px.bar(
                    bar_df, x="Segment", y="Score", color="Emotion",
                    color_discrete_map={k.capitalize(): v for k, v in EMOTION_COLORS.items()},
                    barmode="stack",
                )
                fig_bar.update_layout(
                    height=350,
                    margin=dict(t=20, b=20),
                    xaxis=dict(title="Time Window"),
                    yaxis=dict(title="Score"),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            st.divider()

            # ---- 5. Fine‑Grained Segment Table ----
            st.subheader("📋 Complete Emotion Log (per Whisper segment)")
            st.caption("Each row = one short Whisper segment, classified using surrounding context for accuracy.")

            table_rows = []
            for s in fine_segments:
                emoji = EMOTION_EMOJI.get(s["emotion"], "😶")
                table_rows.append({
                    "Start": _fmt(s["start"]),
                    "End": _fmt(s["end"]),
                    "Emotion": f"{emoji} {s['emotion'].capitalize()}",
                    "Confidence": f"{s['confidence']:.0%}",
                    "Text": s["text"],
                })
            table_df = pd.DataFrame(table_rows)
            st.dataframe(table_df, use_container_width=True, hide_index=True)

            st.divider()

            # ---- 6. Full Transcript ----
            st.subheader("📝 Full Transcript")
            st.write(full_text)

            # ---- 7. Raw JSON ----
            with st.expander("🗂️ View Raw JSON Output"):
                st.json({
                    "duration": round(duration, 2),
                    "segments": segments,
                    "fine_segments": fine_segments,
                    "change_points": changes,
                    "distribution": {k: round(v, 2) for k, v in emo_dur.items()},
                })

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)
