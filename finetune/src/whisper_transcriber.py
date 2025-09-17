#!/usr/bin/env python3
"""
Whisper Transcriber for Audio Processing

Handles audio transcription using the Whisper wpt.pt model for dataset classification training.
"""

from __future__ import annotations

import whisper
import numpy as np
import librosa
import logging
from pathlib import Path
from typing import Optional, Union
import torch

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Handles audio transcription using Whisper model."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the Whisper transcriber.
        
        Args:
            model_path: Path to the Whisper model file (wpt.pt)
            device: Device to run the model on ("auto", "cuda", "cpu")
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model from the specified path."""
        try:
            logger.info(f"Loading Whisper model from {self.model_path}...")
            
            # Check if model file exists
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Whisper model not found at {self.model_path}")
            
            # Load model using the correct API
            import torch
            self.model = whisper.load_model(self.model_path)
            if torch.cuda.is_available() and self.device == "cuda":
                self.model = self.model.cuda()
            logger.info("✓ Whisper model loaded successfully")
            
        except Exception as e:
            logger.error(f"✗ Failed to load Whisper model: {e}")
            raise RuntimeError(f"Failed to load Whisper model: {e}")
    
    def transcribe(
        self, 
        audio: np.ndarray, 
        sample_rate: int,
        language: Optional[str] = None,
        fp16: bool = False
    ) -> str:
        """
        Transcribe audio to text using the same method as the server.
        
        Args:
            audio: Audio array (1D or 2D)
            sample_rate: Sample rate of the audio
            language: Language code for transcription (None for auto-detect)
            fp16: Use half precision for faster inference
            
        Returns:
            Transcribed text
        """
        try:
            if self.model is None:
                raise RuntimeError("Whisper model not loaded")
            
            # Process audio the same way as gt() function
            audio_processed = audio.squeeze().astype(np.float32)
            
            # Resample if needed to 16kHz
            if sample_rate != 16_000:
                logger.debug(f"Resampling from {sample_rate}Hz to 16000Hz")
                audio_processed = librosa.resample(
                    audio_processed, 
                    orig_sr=sample_rate, 
                    target_sr=16_000
                )
            
            # Ensure audio is in the right range
            if np.max(np.abs(audio_processed)) > 1.0:
                audio_processed = audio_processed / np.max(np.abs(audio_processed))
            
            # Transcribe using the same parameters as the server
            result = self.model.transcribe(
                audio_processed, 
                fp16=fp16, 
                language=language,
                verbose=False
            )
            
            if isinstance(result, dict) and "text" in result:
                transcribed_text = result["text"].strip()
                logger.debug(f"Transcribed text: {transcribed_text[:100]}...")
                return transcribed_text
            else:
                logger.warning(f"Unexpected transcription result format: {type(result)}")
                return "Error: Unexpected transcription result format"
                
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return f"Error transcribing audio: {str(e)}"
    
    def transcribe_batch(
        self, 
        audio_list: list[np.ndarray], 
        sample_rates: list[int],
        language: Optional[str] = None,
        fp16: bool = False
    ) -> list[str]:
        """
        Transcribe a batch of audio samples.
        
        Args:
            audio_list: List of audio arrays
            sample_rates: List of sample rates corresponding to audio arrays
            language: Language code for transcription
            fp16: Use half precision for faster inference
            
        Returns:
            List of transcribed texts
        """
        if len(audio_list) != len(sample_rates):
            raise ValueError("Number of audio samples must match number of sample rates")
        
        results = []
        for i, (audio, sr) in enumerate(zip(audio_list, sample_rates)):
            logger.debug(f"Transcribing batch item {i+1}/{len(audio_list)}")
            result = self.transcribe(audio, sr, language, fp16)
            results.append(result)
        
        return results
    
    def get_audio_info(self, audio: np.ndarray, sample_rate: int) -> dict:
        """
        Get information about the audio sample.
        
        Args:
            audio: Audio array
            sample_rate: Sample rate
            
        Returns:
            Dictionary with audio information
        """
        duration = len(audio) / sample_rate
        return {
            "shape": audio.shape,
            "sample_rate": sample_rate,
            "duration": duration,
            "max_amplitude": np.max(np.abs(audio)),
            "rms": np.sqrt(np.mean(audio**2))
        }
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self.model is not None
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_path": self.model_path,
            "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown",
            "model_type": type(self.model).__name__
        }


def create_transcriber(model_path: str, device: str = "auto") -> WhisperTranscriber:
    """
    Create a WhisperTranscriber instance.
    
    Args:
        model_path: Path to the Whisper model file
        device: Device to run the model on
        
    Returns:
        WhisperTranscriber instance
    """
    return WhisperTranscriber(model_path, device)


# Test function
def test_transcriber():
    """Test the transcriber with dummy audio."""
    import tempfile
    import os
    
    # Create dummy audio
    sample_rate = 16000
    duration = 2.0
    frequency = 440.0  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Test transcriber
    model_path = "/home/mbhat/omegalabs-anytoany-bittensor/models/wpt/wpt.pt"
    
    if os.path.exists(model_path):
        transcriber = create_transcriber(model_path)
        result = transcriber.transcribe(audio, sample_rate)
        print(f"Test transcription result: {result}")
        print(f"Model info: {transcriber.get_model_info()}")
    else:
        print(f"Model not found at {model_path}, skipping test")


if __name__ == "__main__":
    test_transcriber()





