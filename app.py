# import streamlit as st
# from transformers import AutoProcessor, Wav2Vec2ForCTC
# import torch
# import librosa
# import os
# from pydub import AudioSegment
# from moviepy.editor import VideoFileClip
# import google.generativeai as genai
# from google import genai
# from google.genai import types

# # ----------- Configuration -----------
# model_id = "facebook/mms-1b-l1107"
# lang_code = "urd-script_arabic"
# api_key = "AIzaSyBEWWn32PxVEaUsoe67GJOEpF4FQT87Kxo"  # ⚠️ Replace with st.secrets for production

# # ----------- Load Processor and Model -----------
# @st.cache_resource
# def load_model_and_processor():
#     processor = AutoProcessor.from_pretrained(model_id, target_lang=lang_code)
#     model = Wav2Vec2ForCTC.from_pretrained(
#         model_id,
#         target_lang=lang_code,
#         ignore_mismatched_sizes=True
#     )
#     model.load_adapter(lang_code)
#     return processor, model

# processor, model = load_model_and_processor()

# # ----------- Audio Conversion -----------
# def get_wav_from_input(file_path, output_path="converted.wav"):
#     ext = os.path.splitext(file_path)[-1].lower()
#     if ext in [".mp4", ".mkv", ".avi", ".mov"]:
#         video = VideoFileClip(file_path)
#         video.audio.write_audiofile(output_path, fps=16000)
#     elif ext in [".mp3", ".aac", ".flac", ".ogg", ".m4a"]:
#         audio = AudioSegment.from_file(file_path)
#         audio = audio.set_frame_rate(16000).set_channels(1)
#         audio.export(output_path, format="wav")
#     elif ext == ".wav":
#         audio = AudioSegment.from_wav(file_path)
#         audio.export(output_path, format="wav")
#     else:
#         raise ValueError("Unsupported file format.")
#     return output_path

# # ----------- Transcription -----------
# def transcribe(file_path):
#     wav_path = get_wav_from_input(file_path)
#     audio, sr = librosa.load(wav_path, sr=16000)
#     inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
#     with torch.no_grad():
#         logits = model(**inputs).logits
#     pred_ids = torch.argmax(logits, dim=-1)
#     return processor.batch_decode(pred_ids)[0]

# # ----------- Gemini Analysis -----------
# def analyze_transcript(transcript):
#     client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

#     system_instr = """
# You are a speech analyst. The following transcription is in Urdu and contains no punctuation — your first task is to correct the transcript by segmenting it into grammatically correct sentences.

# Then:
# 1. Translate the corrected Urdu transcript into English.
# 2. Determine whether the transcript involves a single speaker or multiple speakers.
# 3. If multiple speakers are detected, perform diarization by segmenting the transcript with clear speaker labels.

# ⚠️ Format the segmented transcript *exactly* like this:

# **Segmented Transcript**

# **Urdu:**
# Person 01:
# [Urdu line here]

# Person 02:
# [Urdu line here]

# ...

# **English:**
# Person 01:
# [English line here]

# Person 02:
# [English line here]

# ...

# After that, provide your analysis in the following format:

# **Speaker-wise Analysis**
# [One or two sentences per speaker about tone, emotion, behavior]

# **Sentiment and Communication Style**
# [Concise overall tone: e.g., friendly, formal, tense, etc.]

# **Summary of Discussion**
# [A 2–3 line summary of what the speakers talked about, in English]
# """

#     response = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=[transcript],
#         config=types.GenerateContentConfig(
#             system_instruction=system_instr,
#             temperature=0.0
#         )
#     )
#     return response.text
    
# # def analyze_transcript(transcript: str):
# #     client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# #      system_instr = """
# #         You are a speech analyst. The following transcription is in Urdu and contains no punctuation — your first task is to correct the transcript by segmenting it into grammatically correct sentences.
        
# #         Then:
# #         1. Translate the corrected Urdu transcript into English.
# #         2. Determine whether the transcript involves a single speaker or multiple speakers.
# #         3. If multiple speakers are detected, perform diarization by segmenting the transcript with clear speaker labels.
        
# #         ⚠️ Format the segmented transcript *exactly* like this:
        
# #         **Segmented Transcript**
        
# #         **Urdu:**
# #         Person 01:
# #         [Urdu line here]
        
# #         Person 02:
# #         [Urdu line here]
        
# #         ...
        
# #         **English:**
# #         Person 01:
# #         [English line here]
        
# #         Person 02:
# #         [English line here]
        
# #         ...
        
# #         After that, provide your analysis in the following format:
        
# #         **Speaker-wise Analysis**
# #         [One or two sentences per speaker about tone, emotion, behavior]
        
# #         **Sentiment and Communication Style**
# #         [Concise overall tone: e.g., friendly, formal, tense, etc.]
        
# #         **Summary of Discussion**
# #         [A 2–3 line summary of what the speakers talked about, in English]
# #         """
# #     resp = client.models.generate_content(
# #         model="gemini-2.5-flash",
# #         contents=[transcript],
# #         config=types.GenerateContentConfig(
# #             system_instruction=system_instr,
# #             temperature=0.0
# #         ),
# #     )
# #     return resp.text

# # ----------- Format Display Helper -----------
# def format_transcript_block(text: str) -> str:
#     lines = text.split("Person ")
#     formatted = ""
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         if line.startswith("01:") or line.startswith("02:"):
#             formatted += f"\n**Person {line[:2]}**:\n{line[3:].strip()}\n\n"
#         else:
#             formatted += f"{line.strip()}\n\n"
#     return formatted

# # ----------- Streamlit UI -----------
# # Styled Header
# st.markdown("""
#     <div style="text-align: left; padding-bottom: 1rem;">
#         <h1 style='color:#1f77b4; font-size: 2.5em; font-weight: 800; margin-bottom: 0.2em;'>
#             🎙️ Urdu Audio & Video Speech Analyzer
#         </h1>
#         <p style='color: #CCCCCC; font-size: 1.05em; margin-top: 0;'>
#             Upload Urdu audio or video to get structured transcription, speaker diarization, and smart AI analysis.
#         </p>
#     </div>
# """, unsafe_allow_html=True)

# # File Upload
# st.markdown("### 📂 Upload an audio or video file")
# with st.container():
#     uploaded_file = st.file_uploader(
#         label="",
#         type=["mp3", "mp4", "wav", "mkv", "aac", "ogg", "m4a", "flac"],
#         label_visibility="collapsed"
#     )

# if uploaded_file is not None:
#     with st.spinner("⏳ Transcribing..."):
#         file_name = uploaded_file.name
#         temp_path = f"temp_input{os.path.splitext(file_name)[-1]}"
#         with open(temp_path, "wb") as f:
#             f.write(uploaded_file.read())
#         transcript = transcribe(temp_path)

#     st.markdown("### 📝 Raw Urdu Transcription")
#     st.text(transcript)

#     with st.spinner("🔍 Analyzing with Gemini..."):
#         report = analyze_transcript(transcript)

#     # Extract Segmented Urdu and English
#     segmented_urdu = ""
#     segmented_english = ""
#     analysis_only = ""

#     if "Urdu:" in report and "English:" in report:
#         urdu_start = report.find("Urdu:")
#         english_start = report.find("English:")
#         segmented_urdu = report[urdu_start + len("Urdu:"):english_start].strip()

#         english_section = report[english_start + len("English:"):].strip()
#         if "**Speaker-wise Analysis**" in english_section:
#             parts = english_section.split("**Speaker-wise Analysis**")
#             segmented_english = parts[0].strip()
#             analysis_only = "**Speaker-wise Analysis**" + parts[1].strip()
#         else:
#             segmented_english = english_section.strip()
#             analysis_only = "⚠️ Could not extract structured analysis."

#     # Show Segmented Transcript
#     if segmented_urdu and segmented_english:
#         st.markdown("### 🗣️ Segmented Transcript")
#         col1, col2 = st.columns(2)

#         with col1:
#             st.markdown("####  Urdu")
#             st.markdown(format_transcript_block(segmented_urdu))

#         with col2:
#             st.markdown("####  English")
#             st.markdown(format_transcript_block(segmented_english))

#     # Show Gemini Analysis Only (No transcript repeat)
#     if analysis_only:
#         st.markdown("### 🧠 Gemini Analysis Summary")
#         st.markdown(analysis_only)


import io, os, numpy as np, streamlit as st, librosa, torch, soundfile as sf
from transformers import AutoProcessor, Wav2Vec2ForCTC
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
# from google import genai
# from google.genai import types
import google.generativeai as genai

# ✅ programmatic Start/Stop mic (no WebRTC)
from streamlit_mic_recorder import mic_recorder

# ---------------- Config ----------------
st.set_page_config(page_title="Urdu Speech Analyzer", page_icon="🎙️", layout="wide")
PAGE_TITLE = "🎙️ Urdu Audio & Video Speech Analyzer"
model_id = "facebook/mms-1b-l1107"
lang_code = "urd-script_arabic"
api_key = "AIzaSyBEWWn32PxVEaUsoe67GJOEpF4FQT87Kxo"  # hard-coded as requested

# ---------------- Model ----------------
@st.cache_resource
def load_model_and_processor():
    processor = AutoProcessor.from_pretrained(model_id, target_lang=lang_code)
    model = Wav2Vec2ForCTC.from_pretrained(
        model_id, target_lang=lang_code, ignore_mismatched_sizes=True
    )
    model.load_adapter(lang_code)
    return processor, model

processor, model = load_model_and_processor()

# ---------------- Helpers ----------------
def get_wav_from_input(file_path, output_path="converted.wav"):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext in [".mp4", ".mkv", ".avi", ".mov"]:
        video = VideoFileClip(file_path)
        video.audio.write_audiofile(output_path, fps=16000)
    elif ext in [".mp3", ".aac", ".flac", ".ogg", ".m4a"]:
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
    elif ext == ".wav":
        audio = AudioSegment.from_wav(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
    else:
        raise ValueError("Unsupported file format.")
    return output_path

def save_wav_resampled(audio_f32: np.ndarray, sr_in: int, path: str):
    if sr_in != 16000:
        audio_f32 = librosa.resample(audio_f32, orig_sr=sr_in, target_sr=16000)
    audio_f32 = librosa.util.normalize(audio_f32)
    sf.write(path, audio_f32.astype(np.float32), 16000)

def transcribe(wav_path) -> str:
    audio, sr = librosa.load(wav_path, sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]
def analyze_transcript(transcript: str) -> str:
    # configure once per run
    genai.configure(api_key=api_key)

    system_instr = """
You are a speech analyst. The following transcription is in Urdu and contains no punctuation — your first task is to correct the transcript by segmenting it into grammatically correct sentences.

Then:
1. Translate the corrected Urdu transcript into English.
2. Determine whether the transcript involves a single speaker or multiple speakers.
3. If multiple speakers are detected, perform diarization by segmenting the transcript with clear speaker labels.

⚠️ Format the segmented transcript *exactly* like this:

**Segmented Transcript**

**Urdu:**
Person 01:
[Urdu line here]

Person 02:
[Urdu line here]

...

**English:**
Person 01:
[English line here]

Person 02:
[English line here]

...

After that, provide your analysis in the following format:

**Speaker-wise Analysis**
[One or two sentences per speaker about tone, emotion, behavior]

**Sentiment and Communication Style**
[Concise overall tone: e.g., friendly, formal, tense, etc.]

**Summary of Discussion**
[A 2–3 line summary of what the speakers talked about, in English]
"""

    # Pick a stable, widely-available model name on Streamlit Cloud
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instr
    )

    resp = model.generate_content(
        transcript,
        generation_config={"temperature": 0.0}
    )
    return resp.text

# def analyze_transcript(transcript: str) -> str:
#     client = genai.Client(api_key=api_key)
#     system_instr = """
# You are a speech analyst. The following transcription is in Urdu and contains no punctuation — your first task is to correct the transcript by segmenting it into grammatically correct sentences.

# Then:
# 1. Translate the corrected Urdu transcript into English.
# 2. Determine whether the transcript involves a single speaker or multiple speakers.
# 3. If multiple speakers are detected, perform diarization by segmenting the transcript with clear speaker labels.

# ⚠️ Format the segmented transcript *exactly* like this:

# **Segmented Transcript**

# **Urdu:**
# Person 01:
# [Urdu line here]

# Person 02:
# [Urdu line here]

# ...

# **English:**
# Person 01:
# [English line here]

# Person 02:
# [English line here]

# ...

# After that, provide your analysis in the following format:

# **Speaker-wise Analysis**
# [One or two sentences per speaker about tone, emotion, behavior]

# **Sentiment and Communication Style**
# [Concise overall tone: e.g., friendly, formal, tense, etc.]

# **Summary of Discussion**
# [A 2–3 line summary of what the speakers talked about, in English]
# """
#     resp = client.models.generate_content(
#         model="gemini-2.5-flash",
#         contents=[transcript],
#         config=types.GenerateContentConfig(system_instruction=system_instr, temperature=0.0)
#     )
#     return resp.text

def format_transcript_block(text: str) -> str:
    lines = text.split("Person ")
    out = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("01:") or line.startswith("02:"):
            out += f"\n**Person {line[:2]}**:\n{line[3:].strip()}\n\n"
        else:
            out += f"{line}\n\n"
    return out

# ---------------- Header ----------------
st.markdown(f"""
    <div style="text-align: left; padding-bottom: 1rem;">
        <h1 style='color:#1f77b4; font-size: 2.5em; font-weight: 800; margin-bottom: 0.2em;'>
            {PAGE_TITLE}
        </h1>
        <p style='color: #7c8a98; font-size: 1.05em; margin-top: 0;'>
            Record or upload Urdu speech for structured transcription, diarization, and smart AI analysis.
        </p>
    </div>
""", unsafe_allow_html=True)

# ================= Mic: true Start/Stop + narrow Analyze =================
st.markdown("### 🎤 Live recording")

# The component renders **Start** and **Stop** buttons and keeps recording until you press Stop.
rec = mic_recorder(
    start_prompt="▶️ Start",
    stop_prompt="⏹️ Stop",
    just_once=False,       # allow multiple recordings in a session
    key="recorder",
    format="wav"           # returns WAV bytes
)

# `rec` returns after Stop. Different versions return bytes or a dict — handle both.
audio_bytes, sr_in = None, 44100
if rec is not None:
    if isinstance(rec, dict) and "bytes" in rec:
        audio_bytes = rec["bytes"]
        sr_in = int(rec.get("sample_rate", 44100))
    elif isinstance(rec, (bytes, bytearray)):
        audio_bytes = rec
        sr_in = 44100  # component default
    else:
        # fallback: try to extract .get("audio") etc if lib changes
        audio_bytes = rec.get("audio") if isinstance(rec, dict) else None

if audio_bytes:
    st.success("Audio captured.")
    # Convert to mono float32
    data, sr_read = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr_read:  # prefer the rate embedded in the WAV
        sr_in = sr_read

    # Save as 16 kHz mono for the model
    tmp_wav = "mic_recording.wav"
    save_wav_resampled(data, sr_in, tmp_wav)

    # Minimal playback (no waveform)
    st.audio(audio_bytes, format="audio/wav")
    st.caption(f"Duration: {data.size / sr_in:.2f} s")

    # Slim Analyze button (not full width)
    if st.button("🔍 Analyze", type="primary"):
            with st.spinner("⏳ Transcribing & analyzing..."):
                transcript = transcribe(tmp_wav)     # raw not displayed
                report = analyze_transcript(transcript)

            segmented_urdu = segmented_english = analysis_only = ""
            if "Urdu:" in report and "English:" in report:
                u0 = report.find("Urdu:")
                e0 = report.find("English:")
                segmented_urdu = report[u0 + len("Urdu:"):e0].strip()
                english_section = report[e0 + len("English:"):].strip()
                if "**Speaker-wise Analysis**" in english_section:
                    parts = english_section.split("**Speaker-wise Analysis**")
                    segmented_english = parts[0].strip()
                    analysis_only = "**Speaker-wise Analysis**" + parts[1].strip()
                else:
                    segmented_english = english_section.strip()
                    analysis_only = "⚠️ Could not extract structured analysis."

            if segmented_urdu or segmented_english:
                st.markdown("### 🗣️ Segmented Transcript")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### Urdu")
                    st.markdown(format_transcript_block(segmented_urdu) if segmented_urdu else "_(none)_")
                with c2:
                    st.markdown("#### English")
                    st.markdown(format_transcript_block(segmented_english) if segmented_english else "_(none)_")
            if analysis_only:
                st.markdown("### 🧠 Gemini Analysis Summary")
                st.markdown(analysis_only)

st.markdown("---")

# ================= Upload (unchanged) =================
st.markdown("### 📂 Or upload an audio/video file")
uploaded_file = st.file_uploader(
    label="",
    type=["mp3", "mp4", "wav", "mkv", "aac", "ogg", "m4a", "flac"],
    label_visibility="collapsed"
)
if uploaded_file is not None:
    with st.spinner("⏳ Transcribing..."):
        file_name = uploaded_file.name
        temp_path = f"temp_input{os.path.splitext(file_name)[-1]}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        wav_path = get_wav_from_input(temp_path)
        transcript = transcribe(wav_path)

    with st.spinner("🔍 Analyzing with Gemini..."):
        report = analyze_transcript(transcript)

    segmented_urdu = segmented_english = analysis_only = ""
    if "Urdu:" in report and "English:" in report:
        u0 = report.find("Urdu:")
        e0 = report.find("English:")
        segmented_urdu = report[u0 + len("Urdu:"):e0].strip()
        english_section = report[e0 + len("English:"):].strip()
        if "**Speaker-wise Analysis**" in english_section:
            parts = english_section.split("**Speaker-wise Analysis**")
            segmented_english = parts[0].strip()
            analysis_only = "**Speaker-wise Analysis**" + parts[1].strip()
        else:
            segmented_english = english_section.strip()
            analysis_only = "⚠️ Could not extract structured analysis."

    if segmented_urdu or segmented_english:
        st.markdown("### 🗣️ Segmented Transcript")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Urdu")
            st.markdown(format_transcript_block(segmented_urdu) if segmented_urdu else "_(none)_")
        with c2:
            st.markdown("#### English")
            st.markdown(format_transcript_block(segmented_english) if segmented_english else "_(none)_")
    if analysis_only:
        st.markdown("### 🧠 Gemini Analysis Summary")
        st.markdown(analysis_only)

