import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

def load_config():
    load_dotenv()
    return os.getenv("GOOGLE_API_KEY")

def stream_gemini_response(api_key, prompt, system_instruction=None):
    client = genai.Client(api_key=api_key)
    

    config = None
    if system_instruction:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction
        )
    
    response = client.models.generate_content_stream(
        model="gemini-2.0-flash",
        contents=prompt,
        config=config
    )
    
    for chunk in response:
        print(chunk.text, end="", flush=True)

if __name__ == "__main__":
    api_key = load_config()
    
    if not api_key:
        print("Error: GOOGLE_API_KEY not found in .env file.")
    else:
        user_prompt = "Hello, how are you today?"
        

        sys_instruction = "You are a chat assistant. You are concise with your answers"
        
        print(f"Prompt: {user_prompt}")
        print(f"System Instruction: {sys_instruction}\n---")
        

        stream_gemini_response(api_key, user_prompt, system_instruction=sys_instruction)
