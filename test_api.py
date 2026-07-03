import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

# Replace with exact IDs copied from build.nvidia.com/models
MODELS = {
    "qwen": "qwen/qwen3.5-122b-a10b",       
    "mistral": "mistralai/mistral-small-4-119b-2603", 
    "gptoss": "openai/gpt-oss-20b",            
}

for name, model_id in MODELS.items():
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say hello in one sentence."}],
            max_tokens=600,
        )
        print(f"[{name}] OK -> {resp.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"[{name}] FAILED -> {e}")