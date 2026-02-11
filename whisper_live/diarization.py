"""
Real-time speaker diarization integration using diart.
Based on diart's StreamingInference architecture.
"""

import logging
import threading
import queue
import numpy as np
from typing import Optional, Callable, Dict, List, Any
import time
import os
try:
    from diart import SpeakerDiarization
    from diart.sources import AudioSource
    from diart.inference import StreamingInference
    from pyannote.core import Annotation
    import rx.subject
    import torch
    import torchaudio
    import huggingface_hub
    import pytorch_lightning
    import pytorch_lightning.callbacks.early_stopping
    DIART_AVAILABLE = True
except Exception as e:
    DIART_AVAILABLE = False
    logging.warning(
        f"diart or dependency import failed: {e}. Diarization will be disabled.",
        exc_info=True
    )
    # Placeholders to avoid NameError during module definition and execution
    class AudioSource:
        def __init__(self, *args, **kwargs): pass
    class Annotation:
        def __init__(self, *args, **kwargs): pass
    class SpeakerDiarization:
        def __init__(self, *args, **kwargs): pass
    class StreamingInference:
        def __init__(self, *args, **kwargs): pass

logging.basicConfig(level=logging.INFO)


class WhisperLiveAudioSource(AudioSource):
    """
    Custom AudioSource that receives audio chunks from WhisperLive
    and feeds them to diart's StreamingInference pipeline.
    
    This bridges WhisperLive's audio stream with diart's processing.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 5.0,  # diart default
        step_duration: float = 0.5,   # diart default (500ms)
    ):
        """
        Initialize audio source for diart.
        
        Args:
            sample_rate: Audio sample rate (must match whisper: 16000)
            chunk_duration: Duration of each chunk in seconds
            step_duration: Step between chunks in seconds
        """
        super().__init__(uri="whisperlive-stream", sample_rate=sample_rate)
        self.chunk_duration = chunk_duration
        self.step_duration = step_duration
        
        # Calculate sizes in samples
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.step_samples = int(step_duration * sample_rate)
        
        # Audio buffer for accumulating samples
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_queue = queue.Queue()
        
        # Control flags
        self.is_active = False
        self.is_closed = False
        
        # Stream subject (RxPY) - using simple attribute instead of property
        # to avoid conflicts with diart's base AudioSource class
        self.stream = rx.subject.Subject()
        
        logging.info(
            f"WhisperLiveAudioSource initialized: "
            f"chunk={chunk_duration}s, step={step_duration}s, sr={sample_rate}Hz"
        )

    def add_audio(self, audio_chunk: np.ndarray):
        """
        Add audio chunk from WhisperLive to the processing queue.
        
        Args:
            audio_chunk: Audio samples as numpy array (float32)
        """
        if not self.is_closed:
            self.audio_queue.put(audio_chunk)
    
    def read(self):
        """
        Main read loop - required by AudioSource interface.
        Processes audio from queue and emits chunks to stream.
        """
        self.is_active = True
        logging.info("Diarization AudioSource started reading")
        
        while not self.is_closed:
            try:
                # Get audio with timeout to check is_closed periodically
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Append to buffer
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
                
                # Process complete chunks
                while len(self.audio_buffer) >= self.chunk_samples:
                    # Extract chunk
                    chunk = self.audio_buffer[:self.chunk_samples]
                    
                    # Emit to stream (reshape to (1, samples) for diart)
                    waveform = torch.from_numpy(chunk).reshape(1, -1)
                    self.stream.on_next(waveform)
                    
                    # Slide buffer by step size
                    self.audio_buffer = self.audio_buffer[self.step_samples:]
                    
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in diarization read loop: {e}")
                break
        
        # Cleanup
        self.stream.on_completed()
        self.is_active = False
        logging.info("Diarization AudioSource stopped")
    
    def close(self):
        """Stop the audio source."""
        self.is_closed = True
        logging.info("Diarization AudioSource closing")


class DiarizationManager:
    """
    Manages speaker diarization pipeline for WhisperLive.
    Runs diart's StreamingInference in a separate thread.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        step: float = 0.5,
        latency: float = 0.5,
        tau_active: float = 0.6,
        callback: Optional[Callable[[Annotation], None]] = None,
        hf_token: Optional[str] = None,
    ):
        """
        Initialize diarization manager.
        
        Args:
            sample_rate: Audio sample rate (16000 for Whisper)
            step: Duration of processing step in seconds
            latency: System latency in seconds
            tau_active: Threshold for active speaker detection
            callback: Function called with diarization results (Annotation)
            hf_token: HuggingFace token for pyannote models
        """
        if not DIART_AVAILABLE:
            raise ImportError(
                "diart not available. Install with: pip install diart pyannote.audio"
            )

        if hf_token is not None:
            os.environ["HF_TOKEN"] = hf_token
        elif os.environ.get("HF_TOKEN"):
            hf_token = os.environ.get("HF_TOKEN")
        
        # Login to Hugging Face for model access
        if hf_token:
            huggingface_hub.login(token=hf_token)
            logging.info("Logged in to Hugging Face")
        

        import torch
        import pyannote.audio.core.task
        import pyannote.core.annotation
        
        # Add all safe globals for model loading (required for PyTorch 2.6+)
        # These classes are used by pyannote.audio models during checkpoint loading
        torch.serialization.add_safe_globals([
            pyannote.audio.core.task.Problem,
            pyannote.audio.core.task.Specifications,
            pyannote.audio.core.task.Resolution,
            pyannote.audio.core.task.Task,
            pyannote.core.annotation.Annotation,
            pytorch_lightning.callbacks.early_stopping.EarlyStopping,
        ])
        
        self.sample_rate = sample_rate
        self.callback = callback
        
        # Create audio source
        self.audio_source = WhisperLiveAudioSource(
            sample_rate=sample_rate,
            chunk_duration=5.0,  # diart default
            step_duration=step,
        )
        
        # Initialize diart pipeline
        try:
            from diart import SpeakerDiarizationConfig
            from diart.models import SegmentationModel, EmbeddingModel
            
            # Explicitly load models with token
            segmentation = SegmentationModel.from_pretrained(
                "pyannote/segmentation-3.0",
                # token=hf_token,
            )
            embedding = EmbeddingModel.from_pretrained(
                "pyannote/embedding",
                # token=hf_token,
            )

            config = SpeakerDiarizationConfig(
                segmentation=segmentation,
                embedding=embedding,
                step=step,
                latency=latency,
                tau_active=tau_active,
            )
            self.pipeline = SpeakerDiarization(config)
            logging.info("Diart SpeakerDiarization pipeline initialized with HF token")
        except Exception as e:
            logging.error(f"Failed to initialize diart pipeline: {e}")
            logging.error(
                "Make sure you have accepted pyannote model conditions "
                "and logged in with huggingface-cli"
            )
            raise
        
        # Create streaming inference
        self.inference = StreamingInference(
            pipeline=self.pipeline,
            source=self.audio_source,
            do_plot=False,
        )
        
        # Attach observer for results
        self.inference.attach_hooks(self._on_prediction)
        
        # Threading
        self.inference_thread = None
        self.source_thread = None
        self.running = False
        
        # Store latest results
        self.latest_annotation = None
        self.annotation_lock = threading.Lock()
        
        logging.info("DiarizationManager initialized successfully")
    
    def _on_prediction(self, prediction_tuple):
        """
        Callback for streaming inference predictions.
        
        Args:
            prediction_tuple: (annotation, waveform) tuple from diart
        """
        annotation, waveform = prediction_tuple
        
        with self.annotation_lock:
            self.latest_annotation = annotation
        
        # Call user callback if provided
        if self.callback:
            try:
                self.callback(annotation)
            except Exception as e:
                logging.error(f"Error in diarization callback: {e}")
    
    def start(self):
        """Start diarization processing."""
        if self.running:
            logging.warning("DiarizationManager already running")
            return
        
        self.running = True
        
        # Start audio source read thread
        self.source_thread = threading.Thread(
            target=self.audio_source.read,
            daemon=True,
            name="DiartAudioSource"
        )
        self.source_thread.start()
        
        # Start inference thread
        self.inference_thread = threading.Thread(
            target=self._run_inference,
            daemon=True,
            name="DiartInference"
        )
        self.inference_thread.start()
        
        logging.info("DiarizationManager started")
    
    def _run_inference(self):
        """Run streaming inference (blocking call)."""
        try:
            # This blocks until audio source is closed
            self.inference()
            logging.info("Diarization inference completed")
        except Exception as e:
            logging.error(f"Error in diarization inference: {e}")
    
    def add_audio(self, audio_chunk: np.ndarray):
        """
        Add audio chunk for diarization.
        
        Args:
            audio_chunk: Audio samples (float32 numpy array)
        """
        if self.running:
            self.audio_source.add_audio(audio_chunk)
    
    def get_current_speakers(self, timestamp: float) -> Dict[str, Any]:
        """
        Get active speakers at a specific timestamp.
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            Dictionary with speaker information:
            {
                'speakers': [0, 1],
                'primary_speaker': 0,
                'num_speakers': 2
            }
        """
        with self.annotation_lock:
            if self.latest_annotation is None:
                return {
                    'speakers': [],
                    'primary_speaker': None,
                    'num_speakers': 0
                }
            
            # Find active speakers at timestamp
            speakers = []
            for segment, _, label in self.latest_annotation.itertracks(yield_label=True):
                if segment.start <= timestamp <= segment.end:
                    # Convert 'SPEAKER_00' to 0
                    try:
                        speaker_id = int(label.split('_')[-1])
                        speakers.append(speaker_id)
                    except (ValueError, IndexError):
                        # Fallback if label format is different
                        speakers.append(label)
            
            return {
                'speakers': speakers,
                'primary_speaker': speakers[0] if speakers else None,
                'num_speakers': len(speakers)
            }
    
    def stop(self):
        """Stop diarization processing."""
        if not self.running:
            return
        
        logging.info("Stopping DiarizationManager...")
        self.running = False
        
        # Close audio source
        self.audio_source.close()
        
        # Wait for threads
        if self.source_thread:
            self.source_thread.join(timeout=2.0)
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)
        
        logging.info("DiarizationManager stopped")


def create_diarization_manager(
    sample_rate: int = 16000,
    callback: Optional[Callable] = None,
    **kwargs
) -> Optional[DiarizationManager]:
    """
    Factory function to create DiarizationManager.
    """
    if not DIART_AVAILABLE:
        logging.warning("Cannot create DiarizationManager: diart not available")
        return None
    
    try:
        return DiarizationManager(
            sample_rate=sample_rate,
            callback=callback,
            **kwargs
        )
    except Exception as e:
        logging.error(f"Failed to create DiarizationManager: {e}")
        return None
