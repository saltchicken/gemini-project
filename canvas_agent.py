import os
import re
import argparse
import sys
import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types


# This decouples the parsing from the network handling, prioritizing extraction refactoring.
class CanvasStreamParser:
    """
    Parses a continuous stream of text to separate conversational content
    from file blocks wrapped in <file path="...">...</file>.
    """

    def __init__(self):
        self.buffer = ""
        self.in_file = False
        self.current_file_path = None

        self.open_tag_pattern = re.compile(
            r'<file\s+path=["\']([^"\']+)["\']\s*>', re.DOTALL
        )
        self.close_tag_pattern = re.compile(r"</file>", re.DOTALL)

    def _clean_chat(self, text):
        # Removes standalone ``` or ```xml that might appear as artifacts
        cleaned = re.sub(r"^\s*```\w*\s*$", "", text, flags=re.MULTILINE)
        return cleaned

    def process_chunk(self, chunk):
        """
        Ingests a text chunk and yields events:
        ('chat', text)
        ('file_start', path)
        ('file_content', content)
        ('file_end', path)
        """
        self.buffer += chunk
        events = []

        while True:
            if not self.in_file:
                # We are in chat mode, looking for a file start
                match = self.open_tag_pattern.search(self.buffer)
                if match:

                    # 1. Everything before the tag is chat
                    pre_text = self.buffer[: match.start()]

                    clean_text = self._clean_chat(pre_text)
                    if clean_text:
                        events.append(("chat", clean_text))

                    # 2. Switch state
                    self.in_file = True
                    self.current_file_path = match.group(1)
                    events.append(("file_start", self.current_file_path))

                    # 3. Advance buffer past the tag
                    self.buffer = self.buffer[match.end() :]
                else:
                    # If we see a '<', we wait for more data to confirm if it's a tag.
                    tag_start = self.buffer.find("<")
                    if tag_start != -1:
                        # Flush everything before the potential tag as chat
                        if tag_start > 0:

                            clean_text = self._clean_chat(self.buffer[:tag_start])
                            if clean_text:
                                events.append(("chat", clean_text))
                            self.buffer = self.buffer[tag_start:]
                        # Keep the rest in buffer and wait for next chunk
                        break
                    else:
                        # No potential tag start, flush everything as chat
                        if self.buffer:

                            clean_text = self._clean_chat(self.buffer)
                            if clean_text:
                                events.append(("chat", clean_text))
                            self.buffer = ""
                        break

            else:
                # We are inside a file, looking for the closing tag
                match = self.close_tag_pattern.search(self.buffer)
                if match:

                    # 1. Content before tag is file content
                    file_content = self.buffer[: match.start()]
                    if file_content:
                        events.append(("file_content", file_content))

                    # 2. Switch state
                    events.append(("file_end", self.current_file_path))
                    self.in_file = False
                    self.current_file_path = None

                    # 3. Advance buffer past the tag
                    self.buffer = self.buffer[match.end() :]
                else:
                    # No closing tag yet.

                    tag_start = self.buffer.find("<")
                    if tag_start != -1:
                        # Flush content before the potential tag start
                        if tag_start > 0:
                            events.append(("file_content", self.buffer[:tag_start]))
                            self.buffer = self.buffer[tag_start:]
                        # Wait for more data
                        break
                    else:
                        # Flush all as file content
                        if self.buffer:
                            events.append(("file_content", self.buffer))
                            self.buffer = ""
                        break

        return events

    def flush(self):
        """Returns any remaining text in buffer as chat."""
        if self.buffer and not self.in_file:

            clean_text = self._clean_chat(self.buffer)
            if clean_text:
                return [("chat", clean_text)]
        return []


class GeminiCanvasClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            print("‼️ Error: GOOGLE_API_KEY not found in .env")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

        self.system_prompt = """
        You are an expert coding assistant simulating a 'Canvas' interface.
        
        RULES:
        1. When you write code, you MUST wrap it in the following XML tags:
           <file path="filename.ext">
           ... code content ...
           </file>
        
        2. STRICTLY FORBIDDEN: Do NOT use markdown code blocks (```) or (```xml). Just write the XML tags directly.
        3. You can generate multiple files in one response.
        4. Provide brief explanations outside the file tags.
        """

    def stream_content(self, user_prompt):
        if not self.client:
            return

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            temperature=0.2,  # Lower temperature for better structural adherence
        )

        print("\n>> Requesting Canvas Generation...", flush=True)
        try:
            response = self.client.models.generate_content_stream(
                model="gemini-2.0-flash", contents=user_prompt, config=config
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"\n‼️ API Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Gemini Canvas Agent")
    parser.add_argument("prompt", help="The coding task description")

    parser.add_argument(
        "--out-dir", default="output", help="Base directory to save generated files"
    )

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    generation_dir = os.path.join(args.out_dir, timestamp)

    # Ensure output directory exists
    if not os.path.exists(generation_dir):
        os.makedirs(generation_dir)

    client = GeminiCanvasClient()
    stream_parser = CanvasStreamParser()

    current_file_handle = None

    print(f"Canvas Agent initialized. Output dir: {generation_dir}")
    print(f"Prompt: {args.prompt}\n" + "-" * 50)

    try:

        for text_chunk in client.stream_content(args.prompt):

            events = stream_parser.process_chunk(text_chunk)

            for event_type, data in events:
                if event_type == "chat":
                    # Print chat to stdout immediately
                    sys.stdout.write(data)
                    sys.stdout.flush()

                elif event_type == "file_start":

                    full_path = os.path.join(generation_dir, data)

                    os.makedirs(os.path.dirname(full_path), exist_ok=True)

                    print(
                        f"\n\n\033[93m[Creating File: {data}]\033[0m", end=""
                    )  # Yellow text
                    current_file_handle = open(full_path, "w", encoding="utf-8")

                elif event_type == "file_content":
                    # Write content directly to the file
                    if current_file_handle:
                        current_file_handle.write(data)

                elif event_type == "file_end":

                    if current_file_handle:
                        current_file_handle.close()
                        current_file_handle = None
                    print(f"\n\033[92m[File Saved: {data}]\033[0m\n")  # Green text

        # Flush any remaining buffer
        for event_type, data in stream_parser.flush():
            if event_type == "chat":
                sys.stdout.write(data)

        print("\n" + "-" * 50 + "\nDone.")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        if current_file_handle:
            current_file_handle.close()


if __name__ == "__main__":
    main()
