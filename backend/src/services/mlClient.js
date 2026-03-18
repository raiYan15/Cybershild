import axios from 'axios';
import FormData from 'form-data';

const mlApi = axios.create({
  baseURL: process.env.ML_API_URL || `http://127.0.0.1:${process.env.ML_INTERNAL_PORT || 8000}`,
  timeout: 60000, // voice prediction takes longer
});

export async function predictMessage(text) {
  const { data } = await mlApi.post('/predict', { text });
  return {
    prediction: Number(data.prediction || 0),
    confidence: Number(data.confidence || 0),
    toxicityScore: Number(data.toxicity_score || data.toxicityScore || 0),
    abusiveWords: Array.isArray(data.abusive_words || data.abusiveWords)
      ? (data.abusive_words || data.abusiveWords)
      : [],
    transcription: data.transcription || null,
    emotion: data.emotion || null,
    emotionConfidence: data.emotion_confidence || null,
    aggressionScore: data.aggression_score || null,
    language: data.language || 'english',
    suggestedRewrite: data.suggested_rewrite || null,
  };
}

export async function predictVoice(fileBuffer, originalName) {
  const form = new FormData();
  form.append('file', fileBuffer, originalName || 'audio.wav');
  const { data } = await mlApi.post('/predict_voice', form, {
    headers: { ...form.getHeaders() },
  });
  return mapVoiceResponse(data);
}

export async function predictVoiceText(transcription) {
  const { data } = await mlApi.post('/predict_voice_text', { transcription });
  return mapVoiceResponse(data);
}

function mapVoiceResponse(data) {
  return {
    prediction: Number(data.prediction || 0),
    confidence: Number(data.confidence || 0),
    toxicityScore: Number(data.toxicity_score || data.toxicityScore || 0),
    abusiveWords: Array.isArray(data.abusive_words || data.abusiveWords)
      ? (data.abusive_words || data.abusiveWords)
      : [],
    transcription: data.transcription || '',
    emotion: data.emotion || null,
    emotionConfidence: data.emotion_confidence != null ? Number(data.emotion_confidence) : null,
    aggressionScore: data.aggression_score != null ? Number(data.aggression_score) : null,
    sttSource: data.stt_source || null,
    language: data.language || 'english',
    suggestedRewrite: data.suggested_rewrite || null,
  };
}
