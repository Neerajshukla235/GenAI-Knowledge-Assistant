import requests

OLLAMA_URL = "http://ollama:11434/api/generate"

def generate_with_llm(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 300
                }
            },
            timeout=120
        )

        if response.status_code == 200:
            return response.json().get("response")

    except Exception as e:
        print("Ollama error:", str(e))

    return None
