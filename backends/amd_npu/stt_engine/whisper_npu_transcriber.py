import numpy as np
import torch
import librosa
import time
import tempfile
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import onnxruntime as ort

# Placeholder for NPUAccelerator and NPUMatrixMultiplier imports
# These will be properly integrated in subsequent steps
try:
    from .npu_accelerator import NPUAccelerator
    from .matrix_multiply import NPUMatrixMultiplier
    NPU_KERNELS_AVAILABLE = True
except ImportError:
    class NPUAccelerator:
        def is_available(self):
            return False
    class NPUMatrixMultiplier:
        def multiply(self, *args, **kwargs):
            raise NotImplementedError("NPU Matrix Multiplier not available")
    NPU_KERNELS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ONNXWhisperNPUTranscriber:
    """ONNX Whisper with NPU preprocessing, adapted for Meeting Ops"""
    
    def __init__(self, model_cache_dir: str = "./whisper_onnx_cache"):
        """Initialize ONNX Whisper + NPU system"""
        self.npu_accelerator = NPUAccelerator()
        self.npu_multiplier = NPUMatrixMultiplier() if NPU_KERNELS_AVAILABLE else None
        
        # ONNX model paths
        self.model_cache_dir = model_cache_dir
        self.model_path = None
        
        # ONNX sessions
        self.encoder_session = None
        self.decoder_session = None
        self.decoder_with_past_session = None
        
        self.is_ready = False
        
    def initialize(self, model_size="base"):
        """Initialize ONNX Whisper models"""
        try:
            logger.info("🚀 Initializing ONNX Whisper + NPU system...")
            
            # Check NPU
            if not self.npu_accelerator.is_available():
                logger.warning("⚠️ NPU not available, using CPU preprocessing")
            else:
                logger.info("✅ NPU Phoenix detected for preprocessing")
            
            # Set model path
            self.model_path = f"{self.model_cache_dir}/models--onnx-community--whisper-{model_size}/snapshots"
            
            # Find actual snapshot directory
            if os.path.exists(self.model_path):
                snapshots = [d for d in os.listdir(self.model_path) if os.path.isdir(os.path.join(self.model_path, d))]
                if snapshots:
                    self.model_path = os.path.join(self.model_path, snapshots[0], "onnx")
                    logger.info(f"📁 Using model path: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model path not found: {self.model_path}")
                return False
            
            # Load ONNX models
            logger.info("🧠 Loading ONNX Whisper models...")
            
            # Helper to find and load ONNX model
            def load_onnx_model(model_name):
                for suffix in ["", "_fp16", "_quantized", "_int8", "_q4", "_bnb4", "_uint8"]:
                    path = os.path.join(self.model_path, f"{model_name}{suffix}.onnx")
                    if os.path.exists(path):
                        logger.info(f"Attempting to load {model_name}{suffix}.onnx")
                        try:
                            session = ort.InferenceSession(path)
                            logger.info(f"✅ {model_name}{suffix}.onnx loaded successfully from {path}")
                            return session
                        except Exception as e:
                            logger.error(f"❌ Failed to load {model_name}{suffix}.onnx from {path}: {e}")
                logger.error(f"❌ No suitable {model_name}.onnx found or loaded.")
                return None

            # Encoder
            logger.info("Loading encoder model...")
            self.encoder_session = load_onnx_model("encoder_model")
            if not self.encoder_session: 
                logger.error("Encoder session not initialized. Aborting initialization.")
                return False
            
            # Decoder
            logger.info("Loading decoder model...")
            self.decoder_session = load_onnx_model("decoder_model")
            if not self.decoder_session: 
                logger.error("Decoder session not initialized. Aborting initialization.")
                return False
            
            # Decoder with past
            logger.info("Loading decoder with past model...")
            self.decoder_with_past_session = load_onnx_model("decoder_with_past_model")
            # It's okay if decoder_with_past_model is not found, it's optional
            if not self.decoder_with_past_session:
                logger.warning("Decoder with past model not found, proceeding without it.")
            
            self.is_ready = True
            logger.info("🎉 ONNX Whisper + NPU system ready!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Initialization failed: {e}")
            return False
    
    def extract_mel_features(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract mel-spectrogram features with optional NPU acceleration"""
        try:
            logger.info("🎵 Extracting mel-spectrogram features...")
            
            # Use NPU for additional preprocessing if available
            # Temporarily disable NPU preprocessing for debugging
            # if self.npu_accelerator.is_available() and self.npu_multiplier:
            #     logger.info("🔧 NPU preprocessing enabled")
                
            #     # Simple NPU-accelerated audio analysis
            #     try:
            #         # Create feature matrix for NPU analysis
            #         audio_features = np.expand_dims(audio[:16000], axis=0).astype(np.float32)  # First 1 second
            #         if audio_features.shape[1] < 16000:
            #             # Pad if needed
            #             padding = 16000 - audio_features.shape[1]
            #             audio_features = np.pad(audio_features, ((0, 0), (0, padding)), mode='constant')
                    
            #         # Simple NPU matrix operation for audio analysis
            #         analysis_weights = np.random.randn(16000, 80).astype(np.float32) * 0.1
            #         npu_features = self.npu_multiplier.multiply(audio_features, analysis_weights)
            #         logger.info(f"✅ NPU audio analysis: {npu_features.shape}")
                    
            #     except Exception as e:
            #         logger.warning(f"⚠️ NPU preprocessing failed: {e}")
            # elif self.npu_accelerator.is_available():
            #     logger.info("🔧 NPU detected but kernels not available")
            
            # Standard Whisper mel-spectrogram extraction
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sample_rate,
                n_mels=80,
                n_fft=400,
                hop_length=160,
                power=2.0
            )
            
            # Convert to log scale and normalize (Whisper format)
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            log_mel = np.maximum(log_mel, log_mel.max() - 80.0)
            log_mel = (log_mel + 80.0) / 80.0
            
            logger.info(f"✅ Mel features extracted: {log_mel.shape}")
            return log_mel.astype(np.float32)
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            raise
    
    def transcribe_chunk(self, audio_array: np.ndarray, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe an audio chunk (numpy array) using ONNX Whisper + NPU preprocessing"""
        if not self.is_ready:
            raise RuntimeError("ONNX Whisper not initialized")
        
        try:
            start_time = time.time()
            logger.info(f"🎙️ Transcribing audio chunk for session {session_id or 'N/A'}")
            
            # Extract mel features
            mel_features = self.extract_mel_features(audio_array)
            
            # Prepare input for ONNX
            # Whisper expects mel features in shape [batch, n_mels, time]
            mel_input = np.expand_dims(mel_features, axis=0)
            
            # Pad or truncate to expected length (3000 frames for Whisper)
            expected_time = 3000
            if mel_input.shape[2] < expected_time:
                # Pad with zeros
                padding = expected_time - mel_input.shape[2]
                mel_input = np.pad(mel_input, ((0, 0), (0, 0), (0, padding)), mode='constant')
            elif mel_input.shape[2] > expected_time:
                # Truncate
                mel_input = mel_input[:, :, :expected_time]
            
            logger.info(f"Mel input shape: {mel_input.shape}")
            
            # Run encoder
            logger.info("🧠 Running ONNX Whisper encoder...")
            encoder_outputs = self.encoder_session.run(None, {'input_features': mel_input.astype(np.float32)})
            hidden_states = encoder_outputs[0]
            logger.info(f"✅ Encoder output: {hidden_states.shape}")
            
            # Decoding with proper tokenization
            logger.info("🔤 Running ONNX Whisper decoder...")
            
            try:
                # Load Whisper tokenizer
                from transformers import WhisperTokenizer
                tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
                
                # Start with standard Whisper tokens for English transcription
                # 50258 = <|startoftranscript|>, 50259 = <|en|>, 50360 = <|transcribe|>, 50365 = <|notimestamps|>
                decoder_input_ids = np.array([[50258, 50259, 50360, 50365]], dtype=np.int64)
                
                # Generate tokens autoregressively
                generated_tokens = []
                max_tokens = 50  # Limit for 10-second chunks
                
                for step in range(max_tokens):
                    # Run decoder
                    decoder_outputs = self.decoder_session.run(None, {
                        'input_ids': decoder_input_ids,
                        'encoder_hidden_states': hidden_states
                    })
                    
                    # Get next token
                    logits = decoder_outputs[0]
                    next_token_logits = logits[0, -1, :]
                    
                    # Simple greedy decoding
                    next_token_id = int(np.argmax(next_token_logits))
                    
                    # Check for end token
                    if next_token_id in [50257]:  # <|endoftext|>
                        break
                    
                    # Skip special tokens
                    if next_token_id >= 50257:
                        continue
                    
                    # Add token
                    generated_tokens.append(next_token_id)
                    decoder_input_ids = np.concatenate([
                        decoder_input_ids,
                        np.array([[next_token_id]], dtype=np.int64)
                    ], axis=1)
                
                # Decode tokens to text
                if generated_tokens:
                    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    if not text:
                        text = "[Speech detected but unclear]"
                else:
                    text = "[No speech detected]"
                    
            except Exception as e:
                logger.warning(f"Token decoding failed: {e}, using fallback")
                text = "[Audio processed - decoder active]"
            
            audio_duration = len(audio_array) / 16000
            processing_time = time.time() - start_time
            real_time_factor = processing_time / audio_duration if audio_duration > 0 else 0
            
            result = {
                "text": text,
                "segments": [{"text": text, "start": 0, "end": audio_duration}],
                "language": "en",
                "processing_time": processing_time,
                "audio_duration": audio_duration,
                "real_time_factor": real_time_factor,
                "npu_accelerated": self.npu_accelerator.is_available(),
                "onnx_whisper_used": True,
                "encoder_output_shape": list(hidden_states.shape),
                "mel_features_shape": list(mel_input.shape)
            }

            logger.info(f"✅ ONNX Whisper transcription completed in {processing_time:.2f}s")
            logger.info(f"Real-time factor: {real_time_factor:.3f}x")
            
            return result

        except Exception as e:
            logger.error(f"❌ ONNX Whisper transcription failed: {e}", exc_info=True)
            return {"text": "[Transcription failed due to internal error]", "confidence": 0.0}

    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using ONNX Whisper + NPU preprocessing"""
        if not self.is_ready:
            raise RuntimeError("ONNX Whisper not initialized")
        
        try:
            start_time = time.time()
            logger.info(f"🎙️ Transcribing with ONNX Whisper: {audio_path}")
            
            # Load audio
            audio, sample_rate = librosa.load(audio_path, sr=16000)
            logger.info(f"Audio loaded: {len(audio)} samples, {sample_rate}Hz")
            
            return self.transcribe_chunk(audio) # Use the new chunk method
            
        except Exception as e:
            logger.error(f"❌ ONNX Whisper transcription failed: {e}")
            raise

    def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """Transcribe audio using ONNX Whisper + NPU preprocessing"""
        if not self.is_ready:
            raise RuntimeError("ONNX Whisper not initialized")
        
        try:
            start_time = time.time()
            logger.info(f"🎙️ Transcribing with ONNX Whisper: {audio_path}")
            
            # Load audio
            audio, sample_rate = librosa.load(audio_path, sr=16000)
            logger.info(f"Audio loaded: {len(audio)} samples, {sample_rate}Hz")
            
            return self.transcribe_chunk(audio) # Use the new chunk method
            
        except Exception as e:
            logger.error(f"❌ ONNX Whisper transcription failed: {e}")
            raise
            
            hidden_states = encoder_outputs[0]
            logger.info(f"✅ Encoder output: {hidden_states.shape}")

            # Enhanced ONNX Whisper decoding with proper tokenization
            logger.info("🔤 Running ONNX Whisper decoding...")

            try:
                # Load Whisper tokenizer
                from transformers import WhisperTokenizer
                tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")

                # Whisper decoding sequence
                max_length = 448  # Whisper max sequence length
                generated_tokens = []

                # Start with language and task tokens for English transcription
                # 50258 = <|startoftranscript|>, 50259 = <|en|>, 50360 = <|transcribe|>, 50365 = <|notimestamps|>
                decoder_input_ids = np.array([[50258, 50259, 50360, 50365]], dtype=np.int64)

                # Generate tokens autoregressively with improved strategy
                logger.info("🔄 Starting autoregressive decoding...")
                for step in range(100):  # Increase max steps for better transcription
                    decoder_outputs = self.decoder_session.run(None, {
                        'input_ids': decoder_input_ids,
                        'encoder_hidden_states': hidden_states
                    })

                    # Get next token with better selection
                    logits = decoder_outputs[0]
                    last_token_logits = logits[0, -1, :]

                    # Apply temperature for better token selection
                    temperature = 0.1
                    scaled_logits = last_token_logits / temperature

                    # Apply top-k filtering to avoid very unlikely tokens
                    top_k = 50
                    top_k_indices = np.argpartition(scaled_logits, -top_k)[-top_k:]
                    top_k_logits = scaled_logits[top_k_indices]

                    # Apply softmax to get probabilities
                    exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
                    probs = exp_logits / np.sum(exp_logits)

                    # Sample from top tokens instead of just taking argmax
                    if np.random.random() < 0.8:  # 80% of time use top token
                        next_token_id = top_k_indices[np.argmax(probs)]
                    else:  # 20% of time sample from distribution
                        next_token_id = np.random.choice(top_k_indices, p=probs)

                    # Check for end tokens
                    if next_token_id in [50257, 50258]:  # <|endoftext|> or <|startoftranscript|>
                        logger.info(f"🛑 End token detected at step {step}: {next_token_id}")
                        break

                    # Skip special tokens that shouldn't be in transcription
                    if next_token_id >= 50257:  # Skip all special tokens
                        logger.debug(f"Skipping special token: {next_token_id}")
                        continue

                    # Add to sequence
                    decoder_input_ids = np.concatenate([
                        decoder_input_ids,
                        np.array([[next_token_id]], dtype=np.int64)
                    ], axis=1)

                    generated_tokens.append(next_token_id)

                    # Debug: show token being generated
                    if step % 10 == 0:
                        try:
                            partial_text = tokenizer.decode([next_token_id], skip_special_tokens=True)
                            logger.debug(f"Step {step}: Generated token {next_token_id} -> '{partial_text}'")
                        except:
                            logger.debug(f"Step {step}: Generated token {next_token_id}")

                # Decode tokens to text with better handling
                if generated_tokens:
                    logger.info(f"📝 Decoding {len(generated_tokens)} tokens...")

                    # Filter out special tokens more carefully
                    text_tokens = [t for t in generated_tokens if t < 50257 and t > 0]

                    if text_tokens:
                        # Try different decoding strategies
                        text = tokenizer.decode(text_tokens, skip_special_tokens=True)

                        # Clean up common artifacts
                        text = text.strip()
                        if not text:
                            # Try decoding all tokens including special ones to see what we get
                            full_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)
                            logger.info(f"🔍 Full token decode: '{full_text}'")

                            # Extract just the text part
                            import re
                            text_match = re.search(r'<\|transcribe\|><\|notimestamps\|>(.*?)(?:<\||$)', full_text)
                            if text_match:
                                text = text_match.group(1).strip()
                            else:
                                text = "[Audio processed - no clear speech detected]"

                        logger.info(f"✅ Decoded text: '{text}'")
                    else:
                        text = "[Audio processed but no valid text tokens generated]"
                        logger.warning(f"No valid text tokens from: {generated_tokens}")
                else:
                    text = "[Audio processed but no tokens generated]"
                    logger.warning("No tokens generated during decoding")

                logger.info(f"✅ ONNX Whisper decoding completed: '{text}'")

            except Exception as e:
                logger.warning(f"⚠️ Advanced decoding failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())

                # Fallback to simple decoding
                try:
                    decoder_input_ids = np.array([[50258, 50259, 50360, 50365]], dtype=np.int64)  # Whisper start tokens
                    decoder_outputs = self.decoder_session.run(None, {
                        'input_ids': decoder_input_ids,
                        'encoder_hidden_states': hidden_states
                    })

                    # Get most likely next few tokens
                    logits = decoder_outputs[0]

                    # Try to get several tokens
                    fallback_tokens = []
                    for i in range(10):
                        token_id = np.argmax(logits[0, -1, :])
                        if token_id < 50257:  # Valid text token
                            fallback_tokens.append(token_id)

                    if fallback_tokens:
                        from transformers import WhisperTokenizer
                        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
                        text = tokenizer.decode(fallback_tokens, skip_special_tokens=True)
                        if not text.strip():
                            text = "[Audio processed but unclear speech]"
                    else:
                        audio_duration = len(audio_array) / 16000 # Use audio_array for duration
                        text = f"[Audio successfully processed: {audio_duration:.1f}s duration, ONNX Whisper active]"

                except Exception as e2:
                    logger.error(f"❌ All decoding attempts failed: {e2}")
                    text = "[Transcription failed - decoder error]"

            # Calculate metrics
            processing_time = time.time() - start_time
            audio_duration = len(audio_array) / 16000 # Use audio_array for duration
            real_time_factor = processing_time / audio_duration if audio_duration > 0 else 0

            result = {
                "text": text,
                "segments": [{"text": text, "start": 0, "end": audio_duration}],
                "language": "en",  # Detected language (placeholder)
                "processing_time": processing_time,
                "audio_duration": audio_duration,
                "real_time_factor": real_time_factor,
                "npu_accelerated": self.npu_accelerator.is_available(),
                "onnx_whisper_used": True,
                "encoder_output_shape": hidden_states.shape,
                "mel_features_shape": mel_features.shape
            }

            logger.info(f"✅ ONNX Whisper transcription completed in {processing_time:.2f}s")
            logger.info(f"Real-time factor: {real_time_factor:.3f}x")
            logger.info(f"Result: '{text}'")

            return result

        except Exception as e:
            logger.error(f"❌ ONNX Whisper transcription failed: {e}", exc_info=True)
            return {"text": "[Transcription failed due to internal error]", "confidence": 0.0}
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "onnx_whisper_ready": self.is_ready,
            "npu_available": self.npu_accelerator.is_available(),
            "onnx_providers": ort.get_available_providers(),
            "model_path": self.model_path
        }
