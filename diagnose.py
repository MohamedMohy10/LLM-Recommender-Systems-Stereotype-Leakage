import os, requests, time
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NVIDIA_API_KEY")
url = "https://integrate.api.nvidia.com/v1/chat/completions"
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
payload = {
    "model": "meta/llama-3.3-70b-instruct",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 20,
}

resp = requests.post(url, headers=headers, json=payload, timeout=30)
print("Status:", resp.status_code)
print(resp.text[:300])