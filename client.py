import requests
import sounddevice as sd
import struct
import argparse
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

class GeminiClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            print("‼️ Error: GOOGLE_API_KEY not found in .env file.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

    def generate_text(self, prompt, system_instruction=None):
        if not self.client:
            return None
        
        config = None
        if system_instruction:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        
        print(f"\n>> Gemini Streaming Response:", flush=True)
        full_text = ""
        
        try:

            response = self.client.models.generate_content_stream(
                model="gemini-2.0-flash",
                contents=prompt,
                config=config
            )
            
            for chunk in response:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    full_text += chunk.text
            
            print("\n") # Newline after stream finishes
            return full_text
            
        except Exception as e:
            print(f"\n‼️ Gemini Error: {e}")
            return None

class TTSClient:
    def __init__(self, server_url="http://localhost:8123/tts"):
        self.server_url = server_url
        self.sample_rate = None
        self.header_size = 0

    def _parse_wav_header(self, header_bytes):
        """
        Parses the WAV header to extract sample rate and finding the start of data.
        Returns (sample_rate, data_start_offset)
        """
        try:
            if header_bytes[0:4] != b'RIFF':
                return None, 0
            
            fmt_loc = header_bytes.find(b'fmt ')
            if fmt_loc == -1: 
                return None, 0
                
            sr_offset = fmt_loc + 12
            sample_rate = struct.unpack('<I', header_bytes[sr_offset:sr_offset+4])[0]
            
            data_loc = header_bytes.find(b'data')
            if data_loc == -1:
                return sample_rate, 44
                
            header_size = data_loc + 8 
            
            return sample_rate, header_size
        except Exception as e:
            print(f"Header parse warning: {e}")
            return None, 0

    def stream_audio(self, text, voice=None, temperature=0.9):
        payload = {
            "text": text,
            "temperature": temperature
        }
        
        if voice:
            payload["voice"] = voice

        print(f" >> Sending to TTS (Temp: {temperature}, Voice: {voice or 'Default'}): {text[:50]}...", flush=True)

        try:
            with requests.post(self.server_url, json=payload, stream=True, timeout=10) as response:
                if response.status_code != 200:
                    print(f"Server Error: {response.status_code}")
                    return

                output_stream = None
                first_chunk_buffer = b""
                is_header_processed = False

                print(" >> Connected to TTS. Waiting for audio...", flush=True)

                for chunk in response.iter_content(chunk_size=1024):
                    if not chunk:
                        continue

                    if not is_header_processed:
                        first_chunk_buffer += chunk
                        # Wait for minimal header size
                        if len(first_chunk_buffer) < 44:
                            continue
                        
                        sr, header_len = self._parse_wav_header(first_chunk_buffer)
                        
                        if not sr:
                            print("Could not detect WAV header. Aborting.")
                            return

                        print(f" >> Stream started. Rate: {sr}Hz", flush=True)
                        
                        output_stream = sd.RawOutputStream(
                            samplerate=sr,
                            channels=1,
                            dtype='int16', 
                            blocksize=1024
                        )
                        output_stream.start()

                        # Write the data part of the buffer (skipping header)
                        output_stream.write(first_chunk_buffer[header_len:])
                        
                        is_header_processed = True
                    else:
                        output_stream.write(chunk)

                if output_stream:
                    print(" >> Playback finished.", flush=True)
                    output_stream.stop()
                    output_stream.close()

        except requests.exceptions.Timeout:
            print("Error: Server connection timed out. The model might be stuck generating.")
        except requests.exceptions.ConnectionError:
            print("Could not connect to server. Is it running?")
        except KeyboardInterrupt:
            print("\nStopped.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TTS Streaming Client with Gemini Integration")
    parser.add_argument("text", nargs="?", help="Text to speak OR Prompt for Gemini")
    parser.add_argument("--url", default="http://localhost:8123/tts", help="Server URL")
    parser.add_argument("--temp", type=float, default=0.9, help="Temperature (creativity)")
    parser.add_argument("--voice", type=str, default=None, help="Voice ID (filename in refs/ without extension)")

    parser.add_argument("--gemini", action="store_true", help="Use text argument as a prompt for Gemini")
    
    args = parser.parse_args()


    text_to_process = args.text
    
    if args.gemini and args.text:
        gemini = GeminiClient()
        sys_instruction = "You are a chat assistant. You are concise with your answers"

        text_to_process = gemini.generate_text(args.text, system_instruction=sys_instruction)

    client = TTSClient(server_url=args.url)

    if text_to_process:
        client.stream_audio(text_to_process, voice=args.voice, temperature=args.temp)
    elif not args.gemini: 
        # Default test text (only if no text and no gemini flag provided)
        long_text = (
            "Here is a more comprehensive test to verify the streaming capabilities of your server. "
            "We are sending a significantly larger block of text to ensure that the sentence splitting logic works seamlessly. "
            "By the time you hear this sentence, the GPU should have already finished processing the beginning."
        )
        client.stream_audio(long_text, voice=args.voice, temperature=args.temp)