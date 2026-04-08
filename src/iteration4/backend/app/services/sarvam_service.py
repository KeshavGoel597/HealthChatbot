import requests
import os
import base64
import wave
import io
import re
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()


def _split_text_into_chunks(text: str, max_chars: int) -> List[str]:
    """Split text into chunks at sentence boundaries, each under max_chars."""
    # Split by sentence-ending punctuation (English + Tamil/Hindi)
    sentences = re.split(r'(?<=[.!?।|।\n])\s*', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        # If a single sentence exceeds max_chars, force-split it
        if len(sentence) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            # Force split long sentence
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i:i + max_chars])
            continue

        if len(current_chunk) + len(sentence) + 1 < max_chars:
            current_chunk += (" " + sentence) if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    if not chunks and text.strip():
        chunks = [text.strip()[:max_chars]]

    return chunks


class SarvamService:
    def __init__(self):
        self.api_key = os.getenv("SARVAM_API_KEY")
        self.base_url = "https://api.sarvam.ai"

    def _get_headers(self):
        if not self.api_key:
            print("Warning: SARVAM_API_KEY not found")
            return {}
        return {"api-subscription-key": self.api_key}

    def translate(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Translate text using Sarvam AI, with automatic chunking for long texts."""
        if not self.api_key:
            return text

        if source_lang == target_lang:
            return text

        # Mayura:v1 has a 1000 char limit; chunk accordingly
        chunks = _split_text_into_chunks(text, 900)
        translated_parts = []

        for chunk in chunks:
            if not chunk:
                continue
            payload = {
                "input": chunk,
                "source_language_code": source_lang,
                "target_language_code": target_lang,
                "speaker_gender": "Male",
                "mode": "formal",
                "model": "mayura:v1",
                "enable_preprocessing": True
            }
            try:
                response = requests.post(
                    f"{self.base_url}/translate",
                    json=payload,
                    headers=self._get_headers()
                )
                if not response.ok:
                    print(f"Sarvam Translation Error ({response.status_code}): {response.text[:200]}")
                response.raise_for_status()
                translated = response.json().get("translated_text", chunk)
                translated_parts.append(translated)
            except Exception as e:
                print(f"Sarvam Translation Error: {e}")
                translated_parts.append(chunk)  # Fallback: keep original chunk

        return " ".join(translated_parts) if translated_parts else text

    def text_to_speech(self, text: str, language_code: str) -> Optional[str]:
        """Convert text to speech using Sarvam AI, with automatic chunking."""
        if not self.api_key:
            return None

        # Bulbul has a 500 char limit per input; chunk accordingly
        chunks = _split_text_into_chunks(text, 450)

        audio_results = []
        for chunk in chunks[:8]:  # Limit to 8 chunks
            if not chunk:
                continue
            payload = {
                "inputs": [chunk],
                "target_language_code": language_code,
                "speaker": "aditya",
                "model": "bulbul:v3"
            }
            try:
                response = requests.post(
                    f"{self.base_url}/text-to-speech",
                    json=payload,
                    headers=self._get_headers()
                )
                response.raise_for_status()
                audios = response.json().get("audios", [])
                if audios:
                    audio_results.append(base64.b64decode(audios[0]))
            except Exception as e:
                print(f"TTS chunk error: {e}")

        if not audio_results:
            return None

        if len(audio_results) == 1:
            return base64.b64encode(audio_results[0]).decode('utf-8')

        # Join WAV files
        try:
            output_io = io.BytesIO()
            with wave.open(output_io, 'wb') as wav_out:
                for i, audio_bytes in enumerate(audio_results):
                    with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_in:
                        if i == 0:
                            wav_out.setparams(wav_in.getparams())
                        wav_out.writeframes(wav_in.readframes(wav_in.getnframes()))
            return base64.b64encode(output_io.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error joining audio: {e}")
            return base64.b64encode(audio_results[0]).decode('utf-8')
