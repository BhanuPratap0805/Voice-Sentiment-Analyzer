# Voice Sentiment Analyzer

A web application that analyzes voice sentiment from audio files using OpenAI's Whisper for transcription and a Hugging Face model for emotion classification.

## Features
- **Speech-to-Text**: Converts audio to text using OpenAI Whisper.
- **Emotion Detection**: Classifies emotion in the transcribed text.
- **Excitement Detection**: Identifies excitement based on joy and surprise levels.
- **Visualizations**: Displays emotion distribution and timeline.

## Tech Stack
- **Backend**: Flask (Python)
- **AI/ML**: OpenAI Whisper, Hugging Face Transformers, PyTorch
- **Frontend**: HTML5, CSS3, JavaScript
- **Deployment**: Docker, Gunicorn, Render

## Setup

### Using Docker (Recommended)
This method ensures all system dependencies (like `ffmpeg`) are installed.

1.  Build the image:
    ```bash
    docker build -t voice-sentiment-app .
    ```
2.  Run the container:
    ```bash
    docker run -p 5000:5000 voice-sentiment-app
    ```
3.  Open `http://localhost:5000` in your browser.

### Local Setup
1.  Install system dependencies:
    - **FFmpeg**: Required for Whisper. Install via `apt`, `brew`, or download from [ffmpeg.org](https://ffmpeg.org/).
2.  Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the app:
    ```bash
    python app.py
    ```
