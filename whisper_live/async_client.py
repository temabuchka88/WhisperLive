import asyncio
import json
import logging
import uuid
import websockets
from typing import Optional, Callable, Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncTranscriptionClient:
    """
    A modern asynchronous client for WhisperLive using websockets and asyncio.
    Designed for easy integration into async applications like FastAPI.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 9090,
        lang: Optional[str] = None,
        task: str = "transcribe",
        model: str = "small",
        use_vad: bool = True,
        use_diarization: bool = False,
        send_last_n_segments: int = 10,
        no_speech_thresh: float = 0.45,
        clip_audio: bool = False,
        same_output_threshold: int = 10,
        on_transcription: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.host = host
        self.port = port
        self.lang = lang
        self.task = task
        self.model = model
        self.use_vad = use_vad
        self.use_diarization = use_diarization
        self.send_last_n_segments = send_last_n_segments
        self.no_speech_thresh = no_speech_thresh
        self.clip_audio = clip_audio
        self.same_output_threshold = same_output_threshold
        
        self.on_transcription = on_transcription
        self.uid = str(uuid.uuid4())
        self.websocket = None
        self.is_ready = False
        self._receive_task = None

    async def connect(self):
        """
        Establishes connection to the WhisperLive server and performs handshake.
        """
        uri = f"ws://{self.host}:{self.port}"
        logger.info(f"Connecting to WhisperLive at {uri}...")
        
        try:
            self.websocket = await websockets.connect(uri)
            
            # Handshake
            options = {
                "uid": self.uid,
                "language": self.lang,
                "task": self.task,
                "model": self.model,
                "use_vad": self.use_vad,
                "use_diarization": self.use_diarization,
                "send_last_n_segments": self.send_last_n_segments,
                "no_speech_thresh": self.no_speech_thresh,
                "clip_audio": self.clip_audio,
                "same_output_threshold": self.same_output_threshold,
            }
            await self.websocket.send(json.dumps(options))
            
            # Wait for config/ready message
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data.get("type") == "config":
                logger.info("Connected and ready.")
                self.is_ready = True
                # Start background receiver
                self._receive_task = asyncio.create_task(self._receive_loop())
            else:
                logger.warning(f"Unexpected initial message: {data}")
                
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise

    async def _receive_loop(self):
        """
        Background loop to receive transcription results.
        """
        try:
            async for message in self.websocket:
                data = json.loads(message)
                if self.on_transcription:
                    if asyncio.iscoroutinefunction(self.on_transcription):
                        await self.on_transcription(data)
                    else:
                        self.on_transcription(data)
        except websockets.exceptions.ConnectionClosed:
            logger.info("Connection closed by server.")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
        finally:
            self.is_ready = False

    async def send_audio(self, audio_chunk: bytes):
        """
        Sends PCM audio data to the server.
        """
        if not self.websocket or not self.is_ready:
            logger.error("Client not connected or not ready.")
            return

        try:
            await self.websocket.send(audio_chunk)
        except Exception as e:
            logger.error(f"Error sending audio: {e}")
            self.is_ready = False

    async def close(self):
        """
        Closes the connection.
        """
        if self._receive_task:
            self._receive_task.cancel()
        if self.websocket:
            await self.websocket.close()
            logger.info("Connection closed.")
