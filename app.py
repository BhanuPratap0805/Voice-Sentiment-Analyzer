import os
import tempfile

from flask import Flask, render_template, request, jsonify
import whisper
from transformers import pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# ---------------------------------------------------------------------------
# Load models at startup (first run downloads ~150 MB whisper-base +
# ~260 MB emotion model — subsequent runs use cached weights)
# ---------------------------------------------------------------------------
print("[*] Loading Whisper speech-recognition model (base)...")
asr_model = whisper.load_model("base")

print("[*] Loading emotion classifier...")
emotion_clf = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
)
print("[✓] Models ready.\n")


def _fmt(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _check_excitement(scores: dict[str, float]) -> tuple[bool, float]:
    """Detect excitement from high joy + surprise combination."""
    joy = scores.get("joy", 0)
    surprise = scores.get("surprise", 0)
    if (joy >= 0.25 and surprise >= 0.12) or (surprise >= 0.25 and joy >= 0.12):
        return True, round(joy * 0.6 + surprise * 0.4, 3)
    return False, 0.0


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided."}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    file.save(tmp.name)
    tmp.close()

    try:
        # --- Transcribe with timestamps ---
        result = asr_model.transcribe(tmp.name)
        raw_segments = result.get("segments", [])

        if not raw_segments:
            return jsonify({"error": "No speech detected. Try a clearer recording."}), 400

        duration = raw_segments[-1]["end"]

        # --- Emotion analysis per segment ---
        segments = []
        for seg in raw_segments:
            text = seg["text"].strip()
            if not text:
                continue
            emotions = emotion_clf(text)[0]
            dominant = max(emotions, key=lambda e: e["score"])
            score_dict = {e["label"]: round(e["score"], 3) for e in emotions}

            # Check for excitement (high joy + surprise)
            is_excited, exc_conf = _check_excitement(score_dict)
            if is_excited:
                emo_label = "excitement"
                confidence = exc_conf
                score_dict["excitement"] = exc_conf
            else:
                emo_label = dominant["label"]
                confidence = round(dominant["score"], 3)

            segments.append(
                {
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": text,
                    "emotion": emo_label,
                    "confidence": confidence,
                    "scores": score_dict,
                }
            )

        # --- Detect emotion change-points ---
        changes = []
        for i in range(1, len(segments)):
            if segments[i]["emotion"] != segments[i - 1]["emotion"]:
                changes.append(
                    {
                        "time": segments[i]["start"],
                        "time_fmt": _fmt(segments[i]["start"]),
                        "from": segments[i - 1]["emotion"],
                        "to": segments[i]["emotion"],
                    }
                )

        # --- Emotion distribution (% of total duration) ---
        emo_dur: dict[str, float] = {}
        for seg in segments:
            d = seg["end"] - seg["start"]
            emo_dur[seg["emotion"]] = emo_dur.get(seg["emotion"], 0) + d
        total = sum(emo_dur.values()) or 1
        distribution = {k: round(v / total * 100, 1) for k, v in emo_dur.items()}

        return jsonify(
            {
                "segments": segments,
                "changes": changes,
                "distribution": distribution,
                "duration": round(duration, 2),
                "text": result["text"],
            }
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
