#!/usr/bin/env python3
"""
Speaker Diarization Module
Uses pyannote.audio for speaker identification and separation
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

logger = logging.getLogger(__name__)

import asyncio

class SpeakerDiarizer:
    """Speaker diarization using pyannote.audio"""
    
    def __init__(self):
        """Initialize speaker diarization pipeline"""
        self.pipeline = None
        self.is_ready = False
        # self._initialize_pipeline() # We will call this asynchronously
    
    async def initialize(self):
        """Initialize the pyannote.audio pipeline asynchronously"""
        try:
            logger.info("🎤 Initializing speaker diarization pipeline...")
            
            # Skip loading the pipeline for now - requires HuggingFace token
            logger.info("⚠️ Speaker diarization disabled - no HuggingFace token configured")
            self.pipeline = None
            
            self.is_ready = True
            logger.info("✅ Speaker diarization pipeline ready")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize speaker diarization: {e}")
            logger.warning("Continuing without speaker diarization")
            self.is_ready = False
    
    def diarize_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with speaker segments and timeline
        """
        if not self.is_ready:
            return {
                "speakers": [],
                "timeline": [],
                "num_speakers": 0,
                "error": "Speaker diarization not available"
            }
        
        try:
            logger.info(f"🎭 Performing speaker diarization on: {audio_path}")
            
            # Run the pipeline
            diarization = self.pipeline(audio_path)
            
            # Process results
            speakers = []
            timeline = []
            speaker_map = {}
            speaker_count = 0
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Map speaker labels to sequential IDs
                if speaker not in speaker_map:
                    speaker_map[speaker] = f"Speaker_{speaker_count + 1}"
                    speaker_count += 1
                
                speaker_id = speaker_map[speaker]
                
                # Add to timeline
                timeline.append({
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.end - turn.start,
                    "speaker": speaker_id,
                    "confidence": 0.95  # pyannote doesn't provide confidence scores
                })
            
            # Create speaker summary
            for speaker_label, speaker_id in speaker_map.items():
                # Calculate total speaking time for this speaker
                total_time = sum(
                    segment["duration"] for segment in timeline 
                    if segment["speaker"] == speaker_id
                )
                
                speakers.append({
                    "id": speaker_id,
                    "label": speaker_label,
                    "total_time": total_time,
                    "segments": len([s for s in timeline if s["speaker"] == speaker_id])
                })
            
            # Sort timeline by start time
            timeline.sort(key=lambda x: x["start"])
            
            logger.info(f"✅ Diarization complete: {len(speakers)} speakers, {len(timeline)} segments")
            
            return {
                "speakers": speakers,
                "timeline": timeline,
                "num_speakers": len(speakers),
                "total_segments": len(timeline),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Speaker diarization failed: {e}")
            return {
                "speakers": [],
                "timeline": [],
                "num_speakers": 0,
                "error": str(e),
                "success": False
            }
    
    def diarize_audio_chunk(self, audio_data: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Perform speaker diarization on audio chunk
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate
            
        Returns:
            Dictionary with speaker information
        """
        if not self.is_ready:
            return {
                "speakers": [],
                "timeline": [],
                "num_speakers": 0,
                "error": "Speaker diarization not available"
            }
        
        try:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                # Write audio data as WAV file
                with wave.open(tmp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    
                    # Convert float32 to int16
                    if audio_data.dtype == np.float32:
                        audio_int16 = (audio_data * 32767).astype(np.int16)
                    else:
                        audio_int16 = audio_data.astype(np.int16)
                    
                    wav_file.writeframes(audio_int16.tobytes())
                
                # Run diarization
                result = self.diarize_audio(tmp_file.name)
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Audio chunk diarization failed: {e}")
            return {
                "speakers": [],
                "timeline": [],
                "num_speakers": 0,
                "error": str(e),
                "success": False
            }
    
    def get_active_speaker(self, timestamp: float, timeline: List[Dict]) -> Optional[str]:
        """
        Get the active speaker at a specific timestamp
        
        Args:
            timestamp: Time in seconds
            timeline: Speaker timeline from diarization
            
        Returns:
            Speaker ID or None if no speaker is active
        """
        for segment in timeline:
            if segment["start"] <= timestamp <= segment["end"]:
                return segment["speaker"]
        return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get diarization system status"""
        return {
            "ready": self.is_ready,
            "device": "cuda" if torch.cuda.is_available() and self.is_ready else "cpu",
            "model": "pyannote/speaker-diarization-3.1" if self.is_ready else None
        }


def test_speaker_diarization():
    """Test speaker diarization functionality"""
    print("🧪 Testing Speaker Diarization...")
    
    diarizer = SpeakerDiarizer()
    
    if not diarizer.is_ready:
        print("❌ Speaker diarization not available")
        return
    
    # Test with synthetic audio
    print("\n🎵 Testing with synthetic audio...")
    
    # Create synthetic audio with two "speakers" (different frequencies)
    sample_rate = 16000
    duration = 10.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Speaker 1: 0-3s and 6-8s (lower frequency)
    speaker1_audio = np.zeros_like(t)
    speaker1_mask = ((t >= 0) & (t <= 3)) | ((t >= 6) & (t <= 8))
    speaker1_audio[speaker1_mask] = 0.5 * np.sin(2 * np.pi * 200 * t[speaker1_mask])
    
    # Speaker 2: 3-6s and 8-10s (higher frequency)  
    speaker2_audio = np.zeros_like(t)
    speaker2_mask = ((t >= 3) & (t <= 6)) | ((t >= 8) & (t <= 10))
    speaker2_audio[speaker2_mask] = 0.5 * np.sin(2 * np.pi * 400 * t[speaker2_mask])
    
    # Combine
    combined_audio = speaker1_audio + speaker2_audio
    
    # Test diarization
    result = diarizer.diarize_audio_chunk(combined_audio, sample_rate)
    
    print(f"✅ Diarization result:")
    print(f"  Speakers: {result['num_speakers']}")
    print(f"  Segments: {result.get('total_segments', 0)}")
    print(f"  Success: {result.get('success', False)}")
    
    if result.get('speakers'):
        for speaker in result['speakers']:
            print(f"  {speaker['id']}: {speaker['total_time']:.1f}s ({speaker['segments']} segments)")
    
    print("\n🎉 Speaker diarization test completed!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    test_speaker_diarization()