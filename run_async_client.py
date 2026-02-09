import asyncio
import argparse
import sys
import numpy as np
from whisper_live.async_client import AsyncTranscriptionClient

async def run_client(args):
    """
    Runs the experimental async client.
    """
    def on_transcription(data):
        if data.get("status") == "active_transcription":
            for line in data.get("lines", []):
                speaker_str = f"[Speaker {line['speaker']}] " if line.get('speaker') is not None else ""
                print(f"\r{speaker_str}{line['start']} -> {line['end']}: {line['text']}", end="", flush=True)
                if line.get("completed"):
                    print() # New line for completed segments

    client = AsyncTranscriptionClient(
        host=args.server,
        port=args.port,
        model=args.model,
        lang=args.lang,
        use_diarization=args.use_diarization,
        on_transcription=on_transcription
    )

    try:
        await client.connect()
        
        print(f"Streaming dummy audio (silence) to test connection...")
        print("Press Ctrl+C to stop.")
        
        # Stream some dummy audio (16kHz, 16-bit PCM, 0.5s chunks)
        chunk_size = 16000 # 0.5s at 16000Hz * 2 bytes (int16)
        dummy_chunk = np.zeros(8000, dtype=np.int16).tobytes()
        
        while True:
            await client.send_audio(dummy_chunk)
            await asyncio.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        await client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhisperLive Async Client Demo")
    parser.get_default("server")
    parser.add_argument('--port', '-p', type=int, default=9090, help="Websocket port")
    parser.add_argument('--server', '-s', type=str, default='localhost', help='Server hostname')
    parser.add_argument('--model', '-m', type=str, default='small', help='Model to use')
    parser.add_argument('--lang', '-l', type=str, default='en', help='Language')
    parser.add_argument('--use_diarization', '-d', action='store_true', help='Enable diarization')
    
    args = parser.parse_args()
    asyncio.run(run_client(args))
