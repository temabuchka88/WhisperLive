"""
Real-time speaker diarization integration using diart.
Based on diart's StreamingInference architecture.
"""

import logging
import threading
import queue
import numpy as np
import re
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
    
    # Monkey patch torch.load to use weights_only=False for PyTorch 2.6+
    # This is required because diart/pyannote use torch.load internally
    _original_torch_load = torch.load
    def patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = patched_torch_load
    
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
                
                # Validate audio chunk
                if audio_chunk is None or len(audio_chunk) == 0:
                    logging.warning("Received empty audio chunk, skipping")
                    continue
                
                # Append to buffer
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])
                
                # Process complete chunks
                while len(self.audio_buffer) >= self.chunk_samples:
                    # Extract chunk
                    chunk = self.audio_buffer[:self.chunk_samples]
                    
                    # Emit to stream (reshape to (1, samples) for diart)
                    waveform = torch.from_numpy(chunk).reshape(1, -1)
                    try:
                        self.stream.on_next(waveform)
                    except IndexError as e:
                        if "pop from empty list" in str(e):
                             logging.warning(f"Ignored 'pop from empty list' in diart pipeline. Chunk skipped.")
                        else:
                             logging.error(f"Index error in diart pipeline: {e}")
                    except Exception as stream_error:
                        logging.error(f"Error emitting to stream: {stream_error}", exc_info=True)
                    
                    # Slide buffer by step size
                    if self.step_samples > 0 and len(self.audio_buffer) >= self.step_samples:
                        self.audio_buffer = self.audio_buffer[self.step_samples:]
                    else:
                        # Fallback: clear buffer if step size is invalid
                        self.audio_buffer = np.array([], dtype=np.float32)
                        break
                    
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in diarization read loop: {e}", exc_info=True)
                # Don't break - try to continue processing
                continue
        
        # Cleanup
        try:
            self.stream.on_completed()
        except Exception as e:
            logging.error(f"Error completing stream: {e}")
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
        step: float = 0.3,
        latency: float = 3,
        tau_active: float = 0.5,
        rho_update: float = 0.3,
        delta_new: float = 0.28,
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
            rho_update: How fast speaker embeddings are updated
            delta_new: Threshold for creating a new speaker cluster
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
        

        # Skip safe globals setup - using monkey patch instead
        pass
        
        self.sample_rate = sample_rate
        self.callback = callback
        
        # Create audio source
        self.audio_source = WhisperLiveAudioSource(
            sample_rate=sample_rate,
            chunk_duration=3.0,  # diart default
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
                rho_update=rho_update,
                delta_new=delta_new,
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
        
        # Synchronization for blocking speaker retrieval
        self.annotation_condition = threading.Condition(self.annotation_lock)
        self.last_processed_timestamp = 0.0
        
        # Track total audio submitted to handle silent gaps
        self.total_audio_submitted = 0.0
        
        # Speaker cache for stability - stores last known speaker for fallback
        self._last_known_speaker = None
        
        logging.info("DiarizationManager initialized successfully")
    
    def _on_prediction(self, prediction_tuple):
        """
        Callback for streaming inference predictions.
        
        Args:
            prediction_tuple: (annotation, waveform) tuple from diart
        """
        annotation, waveform = prediction_tuple
        
        with self.annotation_condition:
            self.latest_annotation = annotation
            
            # Update last processed timestamp based on annotation
            # We look for the latest end time in the annotation segments
            max_timestamp = 0.0
            if len(annotation) > 0:
                for segment, _, _ in annotation.itertracks(yield_label=True):
                    if segment.end > max_timestamp:
                        max_timestamp = segment.end
            
            # Update based on annotation if available, otherwise rely on submitted audio
            if max_timestamp > self.last_processed_timestamp:
                self.last_processed_timestamp = max_timestamp
            
            # Notify all waiters that new data is available
            self.annotation_condition.notify_all()
        
        # Log diarization results for debugging
        if len(annotation) > 0:
            # Log all tracks
            unique_speakers = sorted(list(set(label for _, _, label in annotation.itertracks(yield_label=True))))
            num_tracks = len(list(annotation.itertracks()))
            
            if num_tracks > 0:
                logging.debug(f"[DIARIZATION] Annotation trace: {annotation}")
                logging.info(f"[DIARIZATION] Speakers: {unique_speakers}, Tracks: {num_tracks}")
                
                # Log detailed segment info only if there are speakers detected
                for segment, _, label in annotation.itertracks(yield_label=True):
                    logging.info(f"[DIARIZATION] Speaker: {label} | {segment.start:.2f}s - {segment.end:.2f}s")
        
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
            # Update total audio submitted to track progress even during silence
            self.total_audio_submitted += len(audio_chunk) / self.sample_rate
            
            # Also update last_processed_timestamp to track input progress
            # (subtracting a small latency margin to be safe)
            with self.annotation_condition:
                if self.total_audio_submitted > self.last_processed_timestamp:
                    self.last_processed_timestamp = self.total_audio_submitted - self.latency
                    self.annotation_condition.notify_all()
            
            self.audio_source.add_audio(audio_chunk)
    
    def get_current_speakers(self, timestamp: float) -> Dict[str, Any]:
        """
        Get active speakers at a specific timestamp.
        
        This method now includes fallback to last known speaker for better stability.
        
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
        with self.annotation_condition:
            if self.latest_annotation is None:
                # Fallback: use last known speaker
                if hasattr(self, '_last_known_speaker') and self._last_known_speaker is not None:
                    return {
                        'speakers': [self._last_known_speaker],
                        'primary_speaker': self._last_known_speaker,
                        'num_speakers': 1
                    }
                return {
                    'speakers': [],
                    'primary_speaker': None,
                    'num_speakers': 0
                }
            
            # Find active speakers at timestamp
            speakers = []
            for segment, _, label in self.latest_annotation.itertracks(yield_label=True):
                if segment.start <= timestamp <= segment.end:
                    speaker_id = self._parse_speaker_label(label)
                    if speaker_id is not None:
                        speakers.append(speaker_id)
            
            # Cache the last known speaker for fallback
            if speakers:
                self._last_known_speaker = speakers[0]
            
            # If no speakers found at exact timestamp, try to use last known speaker
            # This provides continuity when diarization hasn't processed the exact moment yet
            if not speakers and hasattr(self, '_last_known_speaker') and self._last_known_speaker is not None:
                # Check if we're close to the last known timestamp
                return {
                    'speakers': [self._last_known_speaker],
                    'primary_speaker': self._last_known_speaker,
                    'num_speakers': 1
                }
            
            return {
                'speakers': speakers,
                'primary_speaker': speakers[0] if speakers else None,
                'num_speakers': len(speakers)
            }

    def get_speakers_blocking(self, timestamp: float, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        """
        Get active speakers at a specific timestamp, waiting if necessary.
        
        This method blocks until diarization has processed the audio up to the requested timestamp
        or the timeout is reached.
        
        Args:
            timestamp: Time in seconds (absolute audio time).
            timeout: Maximum time to wait in seconds. Defaults to 2.0. Use <= 0 for infinite wait.
            
        Returns:
            Dictionary with speaker information if available within timeout, otherwise None.
            {
                'speakers': [0, 1],
                'primary_speaker': 0,
                'num_speakers': 2
            }
        """
        start_time = time.time()
        
        with self.annotation_condition:
            # Wait loop
            while self.last_processed_timestamp < timestamp:
                if timeout > 0:
                    remaining_time = timeout - (time.time() - start_time)
                    
                    if remaining_time <= 0:
                        logging.warning(f"[Diarization] Timeout waiting for timestamp {timestamp}. "
                                       f"Processed: {self.last_processed_timestamp}")
                        # Return None on timeout to indicate we should proceed without speakers
                        return None
                    
                    # Wait for new annotation data
                    notified = self.annotation_condition.wait(timeout=remaining_time)
                else:
                    # Infinite wait
                    logging.info(f"[Diarization] Waiting indefinitely for timestamp {timestamp}...")
                    self.annotation_condition.wait()
                
                # Check if we were woken up by new data or just timed out
                # (though wait returns bool, checking processed timestamp is safer)
                if self.last_processed_timestamp >= timestamp:
                    break
            
            # If we are here, we have data or timed out.
            # If timed out (processed < timestamp), we already returned above.
            # Now get speakers using the same logic as get_current_speakers.
            return self.get_current_speakers(timestamp)

    def _parse_speaker_label(self, label: str) -> Optional[int]:
        """
        Parse speaker ID from various label formats returned by diart/pyannote.
        
        Args:
            label: Speaker label string (e.g., 'SPEAKER_00', 'speaker0', 'speaker_01')
        
        Returns:
            Speaker ID as integer, or None if parsing fails
        """
        if not label:
            return None
        
        # Try to extract numeric part from the label using regex
        # Matches patterns like: speaker0, SPEAKER_00, speaker_01, etc.
        match = re.search(r'(\d+)', str(label))
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
        
        # If no digits found, return None
        return None
    
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
