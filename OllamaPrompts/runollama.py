import requests

def query_ollama(prompt, model="llama3.1:8b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["response"]

if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    result = query_ollama(user_prompt)
    print(result)