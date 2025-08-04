from flask import Flask, request, render_template, jsonify
import whisper
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model = whisper.load_model("base")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files['audio']
    filepath = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(filepath)

    result = model.transcribe(filepath)
    transcript = [
        f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}"
        for seg in result["segments"]
    ]

    return jsonify({"transcription": transcript})
