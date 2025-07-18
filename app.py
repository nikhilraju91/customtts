import os, uuid, torch, nltk, numpy as np, soundfile as sf
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from nltk.tokenize import sent_tokenize
from typing import List
import shutil
import requests
from bs4 import BeautifulSoup
import io

# Add the 'src' directory to Python's system path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC  # Added for voice conversion
from eng_to_ipa import convert as ipa_convert

# Download NLTK data
try:
    nltk.download("punkt", quiet=True)
except Exception as e:
    print(f"NLTK download failed: {e}.")

app = FastAPI(
    title="Chatterbox TTS API with Frontend",
    description="A FastAPI application for text-to-speech generation with a simple web UI.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
os.makedirs("audio", exist_ok=True)
os.makedirs("voices", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/audio", StaticFiles(directory="audio"), name="audio")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

try:
    MODEL = ChatterboxTTS.from_pretrained(DEVICE)
    print("Chatterbox TTS model loaded successfully.")
except Exception as e:
    print(f"Failed to load a model: {e}")

@app.get("/", response_class=HTMLResponse)
async def homepage():
    try:
        with open("static/index.html", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found in static directory.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving homepage: {e}")

@app.get("/voices/")
async def list_voices():
    try:
        voice_files = sorted(f for f in os.listdir("voices") if f.endswith(".mp3") or f.endswith(".wav"))
        return JSONResponse(voice_files)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing voices: {e}")

@app.post("/get_ipa/")
async def get_ipa_for_word(word: str = Form(...)):
    import traceback
    print(f"📥 IPA requested for: '{word}'")
    try:
        clean_word = word.strip().split()[0].lower()
        ipa_string = ipa_convert(clean_word).strip()
        if not ipa_string:
            raise ValueError("IPA conversion returned empty result.")
        print(f"✅ IPA output: /{ipa_string}/")
        return JSONResponse({"ipa": ipa_string})
    except Exception as e:
        print("❌ IPA conversion error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to get IPA: {e}")

@app.post("/prepare_text_blocks/")
async def prepare_text_blocks(text: str = Form(""), file: UploadFile = File(None)):
    text_content = ""
    if text.strip():
        text_content = text.strip()
    elif file:
        try:
            content = await file.read()
            text_content = content.decode("utf-8").strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading uploaded file: {e}")
    if not text_content:
        raise HTTPException(status_code=400, detail="No text content found.")
    lines = sent_tokenize(text_content)
    session_id = str(uuid.uuid4())
    os.makedirs(os.path.join("audio", session_id), exist_ok=True)
    return JSONResponse({"session_id": session_id, "lines": lines})

@app.post("/generate/")
async def generate_audio(
    text: str = Form(""),
    split: str = Form("off"),
    voice: str = Form(""),
    exaggeration: float = Form(0.5),
    speed: float = Form(1.0),
    file: UploadFile = File(None)
):
    text_content = ""
    if text.strip():
        text_content = text.strip()
    elif file:
        try:
            content = await file.read()
            text_content = content.decode("utf-8").strip()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading uploaded file: {e}")
    if not text_content:
        raise HTTPException(status_code=400, detail="No input text.")
    lines = sent_tokenize(text_content) if split == "on" else [text_content]
    session_id = str(uuid.uuid4())
    folder = os.path.join("audio", session_id)
    os.makedirs(folder, exist_ok=True)
    voice_path = os.path.join("voices", voice)
    if not os.path.exists(voice_path):
        raise HTTPException(status_code=400, detail=f"Voice prompt '{voice}' not found.")
    clients = []
    try:
        for i, line in enumerate(lines):
            audio = MODEL.generate(
                line,
                audio_prompt_path=voice_path,
                exaggeration=exaggeration,
                temperature=1.0,
                cfg_weight=speed,
                min_p=0.05,
                top_p=1.0,
                repetition_penalty=1.0
            )
            output_path = os.path.join(folder, f"{i}.wav")
            sf.write(output_path, audio.squeeze().numpy(), MODEL.sr)
            clients.append(f"/audio/{session_id}/{i}.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {e}")
    return JSONResponse({"session_id": session_id, "lines": lines, "clients": clients})

@app.post("/regenerate/")
async def regenerate_audio(
    text: str = Form(...),
    voice: str = Form(...),
    exaggeration: float = Form(...),
    speed: float = Form(...),
    session_id: str = Form(...),
    index: int = Form(...)
):
    folder = os.path.join("audio", session_id)
    voice_path = os.path.join("voices", voice)
    output_path = os.path.join(folder, f"{index}.wav")
    if not os.path.exists(voice_path):
        raise HTTPException(status_code=400, detail=f"Voice '{voice}' not found.")
    try:
        audio = MODEL.generate(
            text,
            audio_prompt_path=voice_path,
            exaggeration=exaggeration,
            temperature=1.0,
            cfg_weight=speed,
            min_p=0.05,
            top_p=1.0,
            repetition_penalty=1.0
        )
        sf.write(output_path, audio.squeeze().numpy(), MODEL.sr)
        return JSONResponse({"url": f"/audio/{session_id}/{index}.wav"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio regeneration failed: {e}")

@app.post("/join/{session_id}/")
async def join_audio(session_id: str, ordered_audio_paths: List[str] = Body(..., embed=True)):
    folder = os.path.join("audio", session_id)
    out_path = os.path.join(folder, "joined.wav")
    try:
        segments = []
        sr = None
        for path in ordered_audio_paths:
            abs_path = os.path.abspath(path.lstrip("/"))
            data, cur_sr = sf.read(abs_path)
            if sr is None:
                sr = cur_sr
            elif cur_sr != sr:
                raise HTTPException(status_code=500, detail="Sample rate mismatch.")
            segments.append(np.expand_dims(data, 1) if data.ndim == 1 else data)
        joined = np.concatenate(segments, axis=0)
        sf.write(out_path, joined, sr)
        return JSONResponse({"url": f"/audio/{session_id}/joined.wav"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to join audio files: {e}")

@app.post("/voice_conversion/")
async def voice_conversion(
    target_voice: str = Form(...),
    source_audio: UploadFile = File(...)
):
    session_id = str(uuid.uuid4())
    folder = os.path.join("audio", session_id)
    os.makedirs(folder, exist_ok=True)

    source_path = os.path.join(folder, "source.wav")
    with open(source_path, "wb") as f:
        shutil.copyfileobj(source_audio.file, f)

    target_path = os.path.join("voices", target_voice)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=400, detail=f"Target voice '{target_voice}' not found.")

    try:
        vc_model = ChatterboxVC.from_pretrained(DEVICE)
        converted_audio = vc_model.generate(
            audio=source_path,
            target_voice_path=target_path
        )
        output_path = os.path.join(folder, "converted.wav")
        sf.write(output_path, converted_audio.squeeze().numpy(), vc_model.sr)
        return JSONResponse({"url": f"/audio/{session_id}/converted.wav"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice conversion failed: {e}")

@app.post("/get_merriam_audio/")
async def get_merriam_audio(word: str = Form(...)):
    try:
        dict_url = f"https://www.merriam-webster.com/dictionary/{word}"
        response = requests.get(dict_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        pron_link = soup.find('a', class_='play-pron-v2')
        if not pron_link:
            raise ValueError("Pronunciation link not found.")
        data_dir = pron_link.get('data-dir')
        data_file = pron_link.get('data-file')
        if not data_dir or not data_file:
            raise ValueError("Data attributes not found.")
        mp3_url = f"https://media.merriam-webster.com/audio/prons/en/us/mp3/{data_dir}/{data_file}.mp3"
        mp3_resp = requests.get(mp3_url)
        mp3_resp.raise_for_status()
        session_id = str(uuid.uuid4())
        folder = os.path.join("audio", session_id)
        os.makedirs(folder, exist_ok=True)
        wav_path = os.path.join(folder, "pron.wav")
        audio_data = io.BytesIO(mp3_resp.content)
        data, sr = sf.read(audio_data)
        sf.write(wav_path, data, sr)
        return JSONResponse({"url": f"/audio/{session_id}/pron.wav"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get audio from Merriam-Webster: {str(e)}")

@app.post("/get_pronunciation_suggestions/")
async def get_pronunciation_suggestions(word: str = Form(...)):
    try:
        # Fetch from Dictionary API (free, no key)
        dict_url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word.lower()}"
        response = requests.get(dict_url)
        if response.status_code != 200:
            raise ValueError("Dictionary API request failed.")
        data = response.json()
        ipas = []
        for entry in data:
            for phonetic in entry.get('phonetics', []):
                ipa_text = phonetic.get('text', '').strip('/').strip()
                if ipa_text:
                    ipas.append(ipa_text)
        if not ipas:
            # Fallback to eng_to_ipa
            ipas = [ipa_convert(word.lower()).strip('/').strip()]
        
        # Improved function to generate respellings with more variations
        def ipa_to_respellings(ipa):
            # Expanded mappings with more options for better suggestions
            mappings = {
                'ˈ': ['', ''],
                'ˌ': ['', ''],
                'f': ['f', 'ph'],
                'ɔ': ['aw', 'o', 'or', 'ah'],
                'ɑ': ['ah', 'a', 'ar'],
                'r': ['r', 'rr'],
                'ə': ['uh', 'e', 'a', 'er'],
                'ɪ': ['i', 'ih', 'ee'],
                's': ['s', 'ss'],
                't': ['t', 'tt'],
                '.': ['', ''],  # Syllable separator
                ':': ['', ''],  # Length marker
                'ɹ': ['r', 'rr'],
                'ɛ': ['eh', 'e', 'ai'],
                'ʌ': ['uh', 'u', 'a'],
                'ɒ': ['o', 'aw', 'ah']
            }
            # Generate combinations (limit to avoid explosion)
            respellings = ['']
            for char in ipa.lower():
                if char in mappings:
                    new_respellings = []
                    options = mappings[char]
                    for option in options:
                        for base in respellings:
                            new_respellings.append(base + option)
                    respellings = new_respellings[:len(options) * 2]  # Limit growth
                else:
                    respellings = [base + char for base in respellings]
            # Unique, filter empty, limit to 6, concatenate without hyphens
            unique_respellings = list(set([r for r in respellings if r]))[:6]
            return unique_respellings or [word.lower() + 'ist']  # Fallback
        
        all_respellings = []
        for ipa in ipas:
            all_respellings.extend(ipa_to_respellings(ipa))
        
        unique_suggestions = list(set(all_respellings))[:6]
        if len(unique_suggestions) < 3:
            unique_suggestions.extend([f"{word.lower()}ist", f"{word.lower()}est", f"{word.lower()}uhst"])  # Better generics
        
        return JSONResponse({"suggestions": unique_suggestions})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")
