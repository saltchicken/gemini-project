import os
from dotenv import load_dotenv
from google import genai

def load_config():
    load_dotenv()
    return os.getenv("GOOGLE_API_KEY")

def stream_gemini_response(api_key, prompt):
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents=prompt
    )
    
    for chunk in response:
        print(chunk.text, end="", flush=True)

if __name__ == "__main__":
    api_key = load_config()
    
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file.")
    else:
        user_prompt = "Generate a TidalCycles pattern for a Nu Metal drum beat."
        stream_gemini_response(api_key, user_prompt)
