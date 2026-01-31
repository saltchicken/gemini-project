import requests
import sounddevice as sd
import struct
import argparse
import os
import re
import queue
import threading
from dotenv import load_dotenv
from google import genai
from google.genai import types

# ‼️ New class to handle sentence extraction from a continuous stream
class StreamSentenceSplitter:
    def __init__(self):
        self.buffer = ""
        # Regex to split on punctuation followed by whitespace (looks ahead)
        self.split_pattern = r'(?<=[.!?])\s+'

    def process_chunk(self, text_chunk):
        """Returns a list of complete sentences found in the chunk (plus buffer)."""
        self.buffer += text_chunk
        
        # Try to split
        parts = re.split(self.split_pattern, self.buffer)
        
        if len(parts) > 1:
            # We have at least one split. 
            # All parts except the last one are definitely complete sentences.
            complete_sentences = parts[:-1]
            # The last part is the new buffer (might be incomplete)
            self.buffer = parts[-1]
            return complete_sentences
        else:
            return []

    def flush(self):
        """Returns any remaining text in the buffer."""
        res = self.buffer.strip()
        self.buffer = ""
        return [res] if res else []

# ‼️ New class to manage the Gemini Connection
class GeminiStreamer:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            print("‼️ Error: GOOGLE_API_KEY not found in .env file.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

    def stream_text_generator(self, prompt, system_instruction=None):
        if not self.client:
            return

        config = None
        if system_instruction:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        
        print(f"\n>> Gemini Streaming Request Sent...", flush=True)
        try:
            response = self.client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=prompt,
                config=config
            )
            
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            print(f"\n‼️ Gemini Error: {e}")

# ‼️ Completely refactored Audio Pipeline
class AudioPipeline:
    def __init__(self, server_url, voice=None, temp=0.9):
        self.server_url = server_url
        self.voice = voice
        self.temp = temp
        
        # Queues for the pipeline
        self.sentence_queue = queue.Queue() # Text sentences waiting for TTS
        self.audio_chunk_queue = queue.Queue() # Audio bytes waiting for playback
        
        # Control flags
        self.playback_finished = threading.Event()
        self.tts_processing_finished = threading.Event()
        self.stop_signal = False

        # Audio Config
        self.sample_rate = None
        self.channels = 1
        self.sd_stream = None

    def _parse_wav_header(self, header_bytes):
        """Parses WAV header to get sample rate and data offset."""
        try:
            if len(header_bytes) < 44 or header_bytes[0:4] != b'RIFF':
                return None, 0
            
            fmt_loc = header_bytes.find(b'fmt ')
            if fmt_loc == -1: return None, 0
            
            sr_offset = fmt_loc + 12
            sample_rate = struct.unpack('<I', header_bytes[sr_offset:sr_offset+4])[0]
            
            data_loc = header_bytes.find(b'data')
            if data_loc == -1: return sample_rate, 44
            
            header_size = data_loc + 8 
            return sample_rate, header_size
        except:
            return None, 0

    def tts_worker(self):
        """Thread: Pops sentences, requests TTS, pushes audio chunks."""
        while not self.stop_signal:
            try:
                # Wait for a sentence (timeout allows checking stop_signal)
                text = self.sentence_queue.get(timeout=0.5)
                if text is None: # Sentinel value
                    break
            except queue.Empty:
                continue

            print(f"   [TTS Worker] Processing: '{text[:30]}...'", flush=True)
            
            payload = {
                "text": text,
                "temperature": self.temp,
                "voice": self.voice
            }

            try:
                # ‼️ Send request to server (streaming response)
                with requests.post(self.server_url, json=payload, stream=True, timeout=30) as response:
                    if response.status_code != 200:
                        print(f"‼️ Server Error {response.status_code}")
                        continue
                    
                    for chunk in response.iter_content(chunk_size=4096):
                        if chunk:
                            self.audio_chunk_queue.put(chunk)
            except Exception as e:
                print(f"‼️ TTS Network Error: {e}")

            self.sentence_queue.task_done()
        
        # Signal that no more audio will be produced
        self.tts_processing_finished.set()
        self.audio_chunk_queue.put(None) # Sentinel for player

    def player_worker(self):
        """Thread: Pops audio chunks, plays continuous stream."""
        buffer = b""
        stream_open = False
        
        while not self.stop_signal:
            try:
                chunk = self.audio_chunk_queue.get(timeout=0.5)
                if chunk is None: # Sentinel
                    break
            except queue.Empty:
                continue

            if not stream_open:
                buffer += chunk
                # Need enough bytes for header
                if len(buffer) < 44:
                    continue
                
                # Parse header
                sr, header_len = self._parse_wav_header(buffer)
                if sr:
                    self.sample_rate = sr
                    print(f"   [Player] Stream started at {sr}Hz", flush=True)
                    
                    self.sd_stream = sd.RawOutputStream(
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        dtype='int16', 
                        blocksize=2048
                    )
                    self.sd_stream.start()
                    
                    # Write initial data (skipping header)
                    self.sd_stream.write(buffer[header_len:])
                    buffer = b"" # Clear buffer
                    stream_open = True
                else:
                    # If we can't find header yet, keep buffering
                    continue
            else:
                # ‼️ Crucial: For subsequent sentences, we might get a WAV header again 
                # (because server treats each request as new). We must detect and strip it
                # to avoid loud "pops" or static.
                
                # Simple heuristic: Check if chunk starts with RIFF
                if chunk.startswith(b'RIFF'):
                    _, h_len = self._parse_wav_header(chunk)
                    if h_len > 0:
                        # Strip the header, play the rest
                        self.sd_stream.write(chunk[h_len:])
                    else:
                        self.sd_stream.write(chunk)
                else:
                    self.sd_stream.write(chunk)

            self.audio_chunk_queue.task_done()

        if self.sd_stream:
            self.sd_stream.stop()
            self.sd_stream.close()
        self.playback_finished.set()

    def start(self):
        self.t_tts = threading.Thread(target=self.tts_worker, daemon=True)
        self.t_player = threading.Thread(target=self.player_worker, daemon=True)
        self.t_tts.start()
        self.t_player.start()

    def add_text(self, text):
        self.sentence_queue.put(text)

    def close(self):
        self.sentence_queue.put(None) # Stop TTS
        self.t_tts.join() # Wait for TTS to finish pending
        self.t_player.join() # Wait for player to finish pending

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming Pipeline: Gemini -> TTS -> Audio")
    parser.add_argument("text", nargs="?", help="Initial prompt for Gemini")
    parser.add_argument("--url", default="http://localhost:8123/tts", help="TTS Server URL")
    parser.add_argument("--temp", type=float, default=0.9, help="Audio Temperature")
    parser.add_argument("--voice", type=str, default=None, help="Voice ID")
    parser.add_argument("--gemini", action="store_true", help="Treat input as Gemini Prompt")
    
    args = parser.parse_args()

    pipeline = AudioPipeline(args.url, args.voice, args.temp)
    pipeline.start()

    try:
        if args.gemini and args.text:
            gemini = GeminiStreamer()
            splitter = StreamSentenceSplitter()
            sys_prompt = "You are a conversational assistant. You give lengthy responses."
            
            print(f">> Prompting Gemini: {args.text}")
            
            # ‼️ Main Loop: Stream Gemini -> Split -> Push to Pipeline
            for text_chunk in gemini.stream_text_generator(args.text, system_instruction=sys_prompt):
                print(text_chunk, end="", flush=True) # Print text as it arrives
                sentences = splitter.process_chunk(text_chunk)
                for s in sentences:
                    pipeline.add_text(s)
            
            # Flush remainder
            for s in splitter.flush():
                pipeline.add_text(s)
                
            print("\n>> Gemini Stream Finished. Waiting for audio...")

        elif args.text:
            # Direct text mode (simulating streaming by just pushing it)
            pipeline.add_text(args.text)
        else:
            # Test mode
            test_sentences = [
                "This is the first sentence to test the pipeline.",
                "Here comes the second sentence, which should play immediately after.",
                "Finally, a third sentence to ensure buffering works correctly."
            ]
            for s in test_sentences:
                pipeline.add_text(s)

        pipeline.close()
        print(">> Done.")

    except KeyboardInterrupt:
        print("\nStopping...")
        pipeline.stop_signal = True
        pipeline.close()
