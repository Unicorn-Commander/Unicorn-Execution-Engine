"""
WhisperX NPU Engine - Routes to real NPU implementation
"""

import os

# Check for CPU-only mode
CPU_ONLY_MODE = os.environ.get('CPU_ONLY_MODE', '').lower() in ('1', 'true', 'yes')

if CPU_ONLY_MODE:
    print("🖥️ CPU_ONLY_MODE enabled - using mock WhisperX implementation")
    USE_MOCK = True
else:
    try:
        # Try to import the real NPU engine
        from .whisperx_npu_engine_real import WhisperXNPUEngineReal as WhisperXNPUEngine
        print("✅ Using REAL WhisperX NPU Engine with hardware acceleration")
        USE_MOCK = False
    except ImportError as e:
        print(f"⚠️ Real NPU engine not available: {e}")
        print("⚠️ Falling back to mock implementation")
        USE_MOCK = True

if USE_MOCK:
    
    # Fallback mock implementation
    import numpy as np
    from typing import Dict

    class WhisperXNPUEngine:
        def __init__(self):
            # Initialize with mock NPU acceleration
            self.npu_metrics = {
                "enabled": False,  # Not real NPU
                "speedup_factor": 1,  # No speedup
                "rtf": 1.0,  # Real-time factor
                "device": "CPU (Mock)"
            }
            print(f"Mock WhisperX Engine initialized: {self.npu_metrics}")
        
        def _generate_mock_transcription(self) -> Dict:
            """Generates a mock transcription in the expected format."""
            return {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.5,
                        "text": "Mock transcription - NPU not available.",
                        "speaker": "SPEAKER_00",
                        "confidence": 0.5
                    }
                ],
                "language": "en"
            }

        def transcribe_chunk(self, audio_data: np.ndarray) -> Dict:
            return self._generate_mock_transcription()
        
        def transcribe(self, audio_path: str) -> Dict:
            return self._generate_mock_transcription()
        
        def transcribe_file(self, audio_path: str, diarize: bool = True) -> Dict:
            return self._generate_mock_transcription()
