# streamlit_app.py

import streamlit as st
import whisper
import tempfile
import os

st.title("🎙️ Audio to Text Converter (Whisper)")
st.markdown("Upload an audio file (.mp3/.wav) and get transcribed text with timestamps.")

# Upload audio
audio_file = st.file_uploader("Upload audio file", type=["mp3", "wav", "m4a"])

if audio_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    # Load Whisper
    model = whisper.load_model("base")  # Or 'small', 'medium'

    # Transcribe
    st.info("Transcribing... please wait ⏳")
    result = model.transcribe(temp_audio_path)

    # Show transcription with timestamps
    st.success("Transcription Complete ✅")
    for segment in result["segments"]:
        st.write(f"[{segment['start']:.2f}s - {segment['end']:.2f}s]: {segment['text']}")

    # Optional: Save to TXT
    if st.button("Download as .txt"):
        text_output = "\n".join(
            f"[{s['start']:.2f}s - {s['end']:.2f}s]: {s['text']}" for s in result["segments"]
        )
        st.download_button("Download Transcription", text_output, file_name="transcription.txt")
