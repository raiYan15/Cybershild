from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="CyberShield ML API", version="3.0.0")

# Allow all origins for hackathon convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ML_DIR = Path(__file__).parent

MODEL_CANDIDATES = [
    ML_DIR / "model.pkl",
]
VECTORIZER_CANDIDATES = [
    ML_DIR / "vectorizer.pkl",
]
HINDI_MODEL_CANDIDATES = [
    ML_DIR / "hindi_abuse_model.pkl",
]
HINDI_VECTORIZER_CANDIDATES = [
    ML_DIR / "hindi_vectorizer.pkl",
]

# ─── Globals ──────────────────────────────────────────────────────────────────
model = None
vectorizer = None
hindi_model = None
hindi_vectorizer = None
whisper_model = None
emotion_pipeline = None

# ─── English toxic words ──────────────────────────────────────────────────────
TOXIC_WORDS = {
    "idiot", "stupid", "worthless", "hate", "loser", "dumb", "trash", "ugly",
    "kill", "moron", "shut up", "fool", "pathetic", "nobody likes you",
    "useless", "retard", "bitch", "slut", "nobody wants you", "go die",
    "you suck", "disgusting", "freak", "weirdo", "die",
}

# ─── Hindi toxic words (Devanagari + common romanized) ───────────────────────
# Devanagari terms
HINDI_TOXIC_WORDS_DEVANAGARI = {
    "बेकार", "बेवकूफ", "गधा", "गधे", "कुत्ता", "कुत्ते", "मूर्ख", "निकम्मा",
    "बेशर्म", "कमीना", "हरामी", "हरामज़ादा", "हरामजादा", "साला", "साली",
    "रंडी", "भड़वा", "नालायक", "छिनाल", "चुतिया", "मादरचोद", "भेंचोद",
    "बकवास", "तू मर", "मर जा", "बर्बाद", "घटिया", "सुअर", "सूअर",
    "औलाद", "कमीने", "लोडू", "लौड़ा", "गांडू", "भोसड़ीके", "रंडीबाज",
    "नालायक", "उल्लू", "गाय", "भड़वे", "पागल", "बेहूदा", "बदमाश",
    "धोखेबाज़", "बेईमान", "कायर", "डरपोक",
}

# Common romanized Hindi abuse words/phrases (lowercased, word-boundary matched)
HINDI_TOXIC_ROMANIZED = {
    # Animals as insult
    "gadha", "gadhe", "gadhey", "kutta", "kutte", "suar", "suwar", "suarr",
    "soor", "billi", "chipkali",
    # Aulad / lineage insults
    "aulad", "ki aulad", "suar ki aulad", "suwar ki aulad", "harami ki aulad",
    "kamine ki aulad", "kutte ki aulad", "haramzade ki aulad",
    # Classic abuses
    "kamina", "kamine", "harami", "haramzada", "haramzadi", "haramkhor",
    "haramzade", "haram", "sala", "sali", "saala", "saali",
    "nalayak", "nalaayak", "bewakoof", "bevakoof", "ullu", "ulllu",
    "pagal", "paagal", "bekaar", "nikami", "laanat",
    # Explicit abuses
    "chutiya", "chutiye", "chutiyap", "chut",
    "madarchod", "madar chod", "maa ki", "teri maa", "teri maa ki",
    "bhosdike", "bhosdi", "bhosda", "bhosdiwale",
    "bhenchod", "bhen chod", "teri behen", "bhaand",
    "gandu", "gaandu", "chodu", "chodna", "lodu", "lauda", "lavde",
    "randi", "rundi", "randibaaz", "bhadwa", "bhadwe", "bhadua",
    "dalal", "chhinal", "besharmi", "besharam",
    # Slang abbreviations
    "bsdk", "bc", "mc", "mbc", "lmbc", "saale",
    # Death / violence threats
    "mar ja", "nikal", "bhaag", "mar denge", "jaan se marenge",
    # Misc
    "bakwas", "bakwaas", "jhalla", "ghatiya", "barbad", "barbadi",
    "ghanta", "bakchod", "bakchodi", "maakichut", "lund", "chosna",
    "hijra", "hijda", "napunsak", "chakka",
    "abbe", "oye", "teri", "tere baap", "tera baap",
    "kutte kamino", "kamine log",
}

EMOTION_AGGRESSION_MAP = {
    "anger":    1.00,
    "disgust":  0.85,
    "fear":     0.35,
    "sadness":  0.25,
    "surprise": 0.10,
    "neutral":  0.00,
    "joy":      0.00,
}


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    toxicity_score: float
    abusive_words: List[str]
    language: str = "english"
    suggested_rewrite: Optional[str] = None


class VoiceTextRequest(BaseModel):
    transcription: str = Field(..., min_length=1)


class VoiceAnalysisResponse(BaseModel):
    prediction: int
    confidence: float
    toxicity_score: float
    abusive_words: List[str]
    transcription: str
    emotion: str
    emotion_confidence: float
    aggression_score: float
    stt_source: str
    language: str = "english"
    suggested_rewrite: Optional[str] = None


# ─── Suggested rewrite word maps ────────────────────────────────────────────

# English: abusive phrase → neutral replacement (longest phrases first)
_EN_REPLACEMENTS: dict[str, str] = {
    "nobody likes you":     "you should consider others' feelings",
    "nobody wants you":     "you are valued by someone",
    "go die":               "please leave the conversation",
    "you suck":             "you need to work on this",
    "shut up":              "please be quiet",
    "idiot":                "mistaken person",
    "stupid":               "wrong",
    "worthless":            "struggling",
    "hate":                 "strongly dislike",
    "loser":                "person facing challenges",
    "dumb":                 "uninformed",
    "trash":                "low quality",
    "ugly":                 "unpleasant",
    "kill":                 "stop",
    "moron":                "mistaken person",
    "fool":                 "mistaken person",
    "pathetic":             "disappointing",
    "useless":              "ineffective",
    "retard":               "person with difficulties",
    "bitch":                "person",
    "slut":                 "person",
    "freak":                "unique person",
    "weirdo":               "different person",
    "die":                  "go away",
    "disgusting":           "unpleasant",
}

# Hindi Devanagari: abusive word → neutral replacement
_HI_REPLACEMENTS_DEVANAGARI: dict[str, str] = {
    "बेकार":     "कमज़ोर",
    "बेवकूफ":    "अनजान व्यक्ति",
    "गधा":       "व्यक्ति",
    "गधे":       "व्यक्ति",
    "कुत्ता":    "व्यक्ति",
    "कुत्ते":    "व्यक्ति",
    "मूर्ख":     "अनजान",
    "निकम्मा":   "संघर्षशील",
    "बेशर्म":    "असंवेदनशील",
    "कमीना":     "व्यक्ति",
    "हरामी":     "व्यक्ति",
    "हरामज़ादा": "व्यक्ति",
    "हरामजादा":  "व्यक्ति",
    "साला":      "व्यक्ति",
    "साली":      "व्यक्ति",
    "नालायक":    "असमर्थ",
    "पागल":      "परेशान",
    "बकवास":     "असहमत",
    "घटिया":     "निम्न स्तर",
    "बर्बाद":    "प्रभावित",
    "सुअर":      "व्यक्ति",
    "सूअर":      "व्यक्ति",
    "उल्लू":     "अनजान व्यक्ति",
    "पागल":      "परेशान",
    "बेहूदा":    "अनुचित",
    "बदमाश":     "गलत काम करने वाला",
    "कायर":      "डरा हुआ",
    "डरपोक":     "आशंकित",
    "मर जा":     "दूर जाओ",
    "तू मर":     "बातचीत बंद करो",
}

# Romanized Hindi: abusive → neutral
_HI_REPLACEMENTS_ROMANIZED: dict[str, str] = {
    "gadha": "insaan",
    "gadhe": "insaan",
    "kutta": "insaan",
    "kutte": "insaan",
    "suar": "insaan",
    "kamina": "insaan",
    "harami": "insaan",
    "haramzada": "insaan",
    "bewakoof": "anjaan",
    "bevakoof": "anjaan",
    "ullu": "anjaan",
    "pagal": "pareshan",
    "bakwas": "asahmat",
    "nalayak": "asamarth",
    "mar ja": "door jao",
    "chutiya": "insaan",
    "madarchod": "insaan",
    "bhenchod": "insaan",
    "gandu": "insaan",
    "bsdk": "insaan",
    "bc": "insaan",
    "mc": "insaan",
    "sala": "insaan",
    "saala": "insaan",
    "randi": "insaan",
    "dalal": "insaan",
    "hijra": "vyakti",
    "chakka": "vyakti",
    "ghatiya": "nimnstar",
    "barbad": "prabhavit",
}


def generate_rewrite(text: str, abusive_words: list, language: str) -> Optional[str]:
    """Return a cleaned-up rewrite of `text` with abusive words swapped for neutral ones."""
    if not abusive_words:
        return None

    rewritten = text

    if language == "hindi":
        # 1. Replace Devanagari phrases (longest first)
        sorted_deva = sorted(_HI_REPLACEMENTS_DEVANAGARI.keys(), key=len, reverse=True)
        for phrase in sorted_deva:
            if phrase in rewritten:
                rewritten = rewritten.replace(phrase, _HI_REPLACEMENTS_DEVANAGARI[phrase])
        # 2. Replace romanized phrases case-insensitively (longest first)
        sorted_rom = sorted(_HI_REPLACEMENTS_ROMANIZED.keys(), key=len, reverse=True)
        for phrase in sorted_rom:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            if pattern.search(rewritten):
                rewritten = pattern.sub(_HI_REPLACEMENTS_ROMANIZED[phrase], rewritten)
        # If nothing changed, provide a generic Hindi suggestion
        if rewritten.strip() == text.strip():
            return "अपने विचारों को सम्मानजनक तरीके से व्यक्त करें।"
        return f"सम्मानजनक विकल्प: {rewritten.strip()}"
    else:
        # Replace English phrases (longest first to avoid partial clobbers)
        sorted_en = sorted(_EN_REPLACEMENTS.keys(), key=len, reverse=True)
        for phrase in sorted_en:
            pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
            if pattern.search(rewritten):
                rewritten = pattern.sub(_EN_REPLACEMENTS[phrase], rewritten)
        # Also neutralize any remaining flagged words not in the map
        for word in abusive_words:
            if word not in _EN_REPLACEMENTS:
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                rewritten = pattern.sub("[respectful term]", rewritten)
        if rewritten.strip() == text.strip():
            return "Consider rephrasing to keep the conversation respectful."
        return rewritten.strip()


# ─── Language Detection ───────────────────────────────────────────────────────

# Devanagari Unicode block: U+0900 – U+097F
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')


def detect_language(text: str) -> str:
    """Return 'hindi' if text appears to be Hindi, else 'english'."""
    # 1. Any Devanagari character → definitely Hindi
    if _DEVANAGARI_RE.search(text):
        return "hindi"
    # 2. Check for romanized Hindi abusive keywords
    lower = text.lower()
    for kw in HINDI_TOXIC_ROMANIZED:
        if kw in lower:
            return "hindi"
    return "english"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def extract_abusive_words_english(text: str) -> List[str]:
    lowered = text.lower()
    found = [phrase for phrase in TOXIC_WORDS if phrase in lowered]
    return sorted(set(found))


def extract_abusive_words_hindi(text: str) -> List[str]:
    """Extract Hindi abusive words — both Devanagari and romanized forms.
    Returns longest matching phrases first, skipping sub-phrases already
    covered by a longer match (e.g. 'suar' is suppressed when 'suar ki aulad' matched).
    """
    found_raw: List[str] = []

    # 1. Devanagari — exact substring match
    for phrase in HINDI_TOXIC_WORDS_DEVANAGARI:
        if phrase in text:
            found_raw.append(phrase)

    # 2. Romanized — sort longest-first so compound phrases are preferred
    lower = text.lower()
    sorted_phrases = sorted(HINDI_TOXIC_ROMANIZED, key=len, reverse=True)
    for phrase in sorted_phrases:
        if phrase in lower:
            found_raw.append(phrase)

    # 3. Deduplicate: remove any entry that is a sub-string of another entry
    found_raw = sorted(set(found_raw), key=len, reverse=True)
    deduped: List[str] = []
    for candidate in found_raw:
        if not any(candidate in longer for longer in deduped):
            deduped.append(candidate)

    return sorted(deduped)



def heuristic_predict_english(text: str):
    abusive_words = extract_abusive_words_english(text)
    exclamations = text.count("!")
    uppercase_ratio = (sum(1 for c in text if c.isupper()) / len(text)) if text else 0
    signal = len(abusive_words) * 0.22 + exclamations * 0.02 + uppercase_ratio * 0.4
    toxicity = float(np.clip(signal, 0, 1))
    prediction = 1 if toxicity >= 0.35 else 0
    confidence = float(np.clip(0.55 + toxicity * 0.4, 0.5, 0.99))
    return prediction, confidence, toxicity, abusive_words


def heuristic_predict_hindi(text: str):
    abusive_words = extract_abusive_words_hindi(text)
    exclamations = text.count("!")
    signal = len(abusive_words) * 0.30 + exclamations * 0.02
    toxicity = float(np.clip(signal, 0, 1))
    prediction = 1 if toxicity >= 0.30 else 0
    confidence = float(np.clip(0.55 + toxicity * 0.4, 0.5, 0.99))
    return prediction, confidence, toxicity, abusive_words


def ml_predict(text: str):
    """Route to Hindi or English model based on detected language. Returns (prediction, confidence, toxicity_score, abusive_words, language)."""
    lang = detect_language(text)

    if lang == "hindi":
        abusive_words = extract_abusive_words_hindi(text)
        if hindi_model is not None and hindi_vectorizer is not None and text.strip():
            try:
                x = hindi_vectorizer.transform([text])
                prediction = int(hindi_model.predict(x)[0])
                if hasattr(hindi_model, "predict_proba"):
                    confidence = float(np.max(hindi_model.predict_proba(x)[0]))
                elif hasattr(hindi_model, "decision_function"):
                    dist = float(hindi_model.decision_function(x)[0])
                    prob = 1.0 / (1.0 + np.exp(-dist))
                    confidence = float(prob if prediction == 1 else 1.0 - prob)
                else:
                    confidence = 0.82
                base_toxicity = confidence if prediction == 1 else 1.0 - confidence
                word_penalty = min(0.35, len(abusive_words) * 0.15)
                if prediction == 1:
                    toxicity_score = float(np.clip(base_toxicity + word_penalty, 0.5, 0.99))
                else:
                    toxicity_score = float(np.clip(base_toxicity + word_penalty, 0.0, 0.49))
                return prediction, confidence, toxicity_score, abusive_words, "hindi"
            except Exception:
                pass
        # Hindi heuristic fallback
        prediction, confidence, toxicity_score, abusive_words = heuristic_predict_hindi(text)
        return prediction, confidence, toxicity_score, abusive_words, "hindi"

    # ── English path ────────────────────────────────────────────────────────
    if model is not None and vectorizer is not None and text.strip():
        try:
            x = vectorizer.transform([text])
            prediction = int(model.predict(x)[0])
            if hasattr(model, "predict_proba"):
                confidence = float(np.max(model.predict_proba(x)[0]))
            elif hasattr(model, "decision_function"):
                dist = float(model.decision_function(x)[0])
                prob = 1.0 / (1.0 + np.exp(-dist))
                confidence = float(prob if prediction == 1 else 1.0 - prob)
            else:
                confidence = 0.82
            abusive_words = extract_abusive_words_english(text)
            base_toxicity = confidence if prediction == 1 else 1.0 - confidence
            word_penalty = min(0.35, len(abusive_words) * 0.15)
            if prediction == 1:
                toxicity_score = float(np.clip(base_toxicity + word_penalty, 0.5, 0.99))
            else:
                toxicity_score = float(np.clip(base_toxicity + word_penalty, 0.0, 0.49))
            return prediction, confidence, toxicity_score, abusive_words, "english"
        except Exception:
            pass
    prediction, confidence, toxicity_score, abusive_words = heuristic_predict_english(text)
    return prediction, confidence, toxicity_score, abusive_words, "english"


def analyse_emotion(text: str):
    """Return (emotion_label, emotion_conf, aggression_score)."""
    if emotion_pipeline and text.strip():
        try:
            results = emotion_pipeline(text[:512])
            if results:
                best = results[0]
                lbl = best["label"].lower()
                conf = float(best["score"])
                agg = EMOTION_AGGRESSION_MAP.get(lbl, 0.0) * conf
                return lbl, conf, agg
        except Exception:
            pass
    return "neutral", 1.0, 0.0


def _pick_existing_path(candidates: List[Path]) -> Path | None:
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def _build_voice_response(transcription: str, stt_source: str) -> VoiceAnalysisResponse:
    """Core pipeline: transcription → language detection → emotion → toxicity → combined score."""
    emotion, emotion_conf, aggression_score = analyse_emotion(transcription)
    clean_text = re.sub(r"\s+", " ", transcription).strip()
    prediction, confidence, toxicity_score, abusive_words, language = ml_predict(clean_text)

    final_toxicity = float(np.clip(
        (toxicity_score * 0.7) + (aggression_score * 0.3), 0.0, 0.99
    ))
    if final_toxicity >= 0.5:
        prediction = 1

    rewrite = generate_rewrite(clean_text, abusive_words, language) if prediction == 1 else None

    return VoiceAnalysisResponse(
        prediction=prediction,
        confidence=confidence,
        toxicity_score=final_toxicity,
        abusive_words=abusive_words,
        transcription=transcription,
        emotion=emotion,
        emotion_confidence=emotion_conf,
        aggression_score=aggression_score,
        stt_source=stt_source,
        language=language,
        suggested_rewrite=rewrite,
    )


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def load_artifacts() -> None:
    global model, vectorizer, hindi_model, hindi_vectorizer, whisper_model, emotion_pipeline

    # 1. English SVM / LR model
    try:
        model_path = _pick_existing_path(MODEL_CANDIDATES)
        vectorizer_path = _pick_existing_path(VECTORIZER_CANDIDATES)
        if model_path and vectorizer_path:
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            print(f"[CyberShield] ✓ English ML model loaded: {model_path.name}")
        else:
            print("[CyberShield] ⚠ English ML model not found – using heuristic fallback")
    except Exception as exc:
        print(f"[CyberShield] ✗ English ML model load error: {exc}")

    # 2. Hindi abuse model
    try:
        hindi_model_path = _pick_existing_path(HINDI_MODEL_CANDIDATES)
        hindi_vec_path = _pick_existing_path(HINDI_VECTORIZER_CANDIDATES)
        if hindi_model_path and hindi_vec_path:
            hindi_model = joblib.load(hindi_model_path)
            hindi_vectorizer = joblib.load(hindi_vec_path)
            print(f"[CyberShield] ✓ Hindi ML model loaded: {hindi_model_path.name}")
        else:
            print("[CyberShield] ⚠ Hindi ML model not found – using heuristic fallback for Hindi")
    except Exception as exc:
        print(f"[CyberShield] ✗ Hindi ML model load error: {exc}")

    # 3. OpenAI Whisper
    try:
        import whisper
        whisper_model = whisper.load_model("tiny")
        print("[CyberShield] ✓ Whisper STT loaded (tiny)")
    except Exception as exc:
        whisper_model = None
        print(f"[CyberShield] ⚠ Whisper unavailable: {exc}")

    # 4. HuggingFace Emotion classifier
    try:
        from transformers import pipeline
        emotion_pipeline = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=False,
        )
        print("[CyberShield] ✓ Emotion pipeline loaded")
    except Exception as exc:
        emotion_pipeline = None
        print(f"[CyberShield] ⚠ Emotion pipeline unavailable: {exc}")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "ml_model": bool(model and vectorizer),
        "hindi_model": bool(hindi_model and hindi_vectorizer),
        "whisper": bool(whisper_model),
        "emotion": bool(emotion_pipeline),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    text = re.sub(r"\s+", " ", payload.text).strip()
    prediction, confidence, toxicity_score, abusive_words, language = ml_predict(text)
    rewrite = generate_rewrite(text, abusive_words, language) if prediction == 1 else None
    return PredictResponse(
        prediction=prediction,
        confidence=confidence,
        toxicity_score=toxicity_score,
        abusive_words=abusive_words,
        language=language,
        suggested_rewrite=rewrite,
    )


# ── Path A: frontend sends raw audio → Whisper transcribes on server ──────────
@app.post("/predict_voice", response_model=VoiceAnalysisResponse)
async def predict_voice(file: UploadFile = File(...)):
    """Accept audio upload, transcribe with Whisper, return full analysis."""
    temp_path = ""
    stt_source = "none"

    try:
        suffix = Path(file.filename or "audio.webm").suffix or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        transcription = ""
        if whisper_model:
            try:
                result = whisper_model.transcribe(temp_path, fp16=False)
                transcription = result.get("text", "").strip()
                stt_source = "whisper"
            except Exception as exc:
                print(f"[Whisper] transcription error: {exc}")

        if not transcription:
            transcription = "No speech detected. Please speak clearly and try again."
            stt_source = "none"

        return _build_voice_response(transcription, stt_source)

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ── Path B: frontend already has transcript (Web Speech API) → skip Whisper ──
@app.post("/predict_voice_text", response_model=VoiceAnalysisResponse)
def predict_voice_text(payload: VoiceTextRequest) -> VoiceAnalysisResponse:
    """
    Called when the browser's Web Speech API has already produced a transcript.
    We detect the language, then run emotion + toxicity analysis.
    """
    transcription = re.sub(r"\s+", " ", payload.transcription).strip()
    return _build_voice_response(transcription, "web_speech")
