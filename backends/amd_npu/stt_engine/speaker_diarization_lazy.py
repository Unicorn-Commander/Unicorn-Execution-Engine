#!/usr/bin/env python3
"""
Lazy-loaded Speaker Diarization Module
Only loads GPU resources when actually needed
"""

import torch
import numpy as np
import tempfile
import os
import logging
from typing import List, Dict, Tuple, Optional, Any
from pyannote.audio import Pipeline
from pyannote.core import Segment, Annotation
import wave
import asyncio

logger = logging.getLogger(__name__)

class LazyDiarizer:
    """Lazy-loaded speaker diarization - only uses GPU when needed"""
    
    def __init__(self):
        """Initialize without loading models"""
        self.pipeline = None
        self.is_ready = False
        self._is_loading = False
        logger.info("🎤 LazyDiarizer initialized (models not loaded)")
    
    async def _ensure_loaded(self):
        """Load models only when needed"""
        if self.is_ready or self._is_loading:
            return
            
        self._is_loading = True
        try:
            logger.info("🎤 Loading speaker diarization pipeline on demand...")
            
            # Load the pipeline
            self.pipeline = await asyncio.to_thread(
                Pipeline.from_pretrained,
                "pyannote/speaker-diarization-3.1",
                use_auth_token=None  # Set to your HF token if needed
            )
            
            # Configure the pipeline
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to(torch.device("cuda"))
                logger.info("✅ Using CUDA for speaker diarization")
            else:
                logger.info("⚠️ Using CPU for speaker diarization")
            
            self.is_ready = True
            logger.info("✅ Speaker diarization pipeline ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to load speaker diarization pipeline: {e}")
            self._is_loading = False
            raise
        finally:
            self._is_loading = False
    
    async def process_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, List[Tuple[float, float]]]:
        """Process audio data and return speaker segments - loads models if needed"""
        # Ensure models are loaded
        await self._ensure_loaded()
        
        if not self.is_ready:
            logger.warning("Speaker diarization pipeline not ready")
            return {}
        
        try:
            # Create temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # Write WAV file
                with wave.open(tmp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)   # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
                
                # Process with pipeline
                diarization = self.pipeline(tmp_file.name)
                
                # Extract speaker segments
                speakers = {}
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if speaker not in speakers:
                        speakers[speaker] = []
                    speakers[speaker].append((turn.start, turn.end))
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return speakers
                
        except Exception as e:
            logger.error(f"Error in speaker diarization: {e}")
            return {}
    
    def unload(self):
        """Unload models to free GPU memory"""
        if self.pipeline is not None:
            logger.info("📤 Unloading speaker diarization models from GPU")
            del self.pipeline
            self.pipeline = None
            self.is_ready = False
            
            # Force GPU memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Global instance - but won't load models until used
lazy_diarizer = LazyDiarizer()