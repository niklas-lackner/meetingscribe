
import copy
import sys
from pathlib import Path

import requests
import urllib3

# Suppress only the single InsecureRequestWarning from urllib3 needed for self-signed certs
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- Configuration ---
API_BASE_URL = "https://chat.ai.diz.uk-erlangen.de/api"
MODELS_TO_TEST = [
    "mistral-large:123b",
    "llama3.3:70b",
    "gemma4:31b",
    "medgemma:27b",
]
# --- End Configuration ---


def get_api_key() -> str:
    """Reads the API key from apikey.txt."""
    try:
        apikey_file = Path(__file__).parent / "apikey.txt"
        if apikey_file.exists():
            return apikey_file.read_text(encoding="utf-8-sig").strip()
    except Exception as e:
        print(f"Could not read apikey.txt: {e}", file=sys.stderr)
    return ""


def run_test(api_key: str):
    """Runs a connection test for each model."""
    if not api_key:
        print("API key is missing. Please create 'apikey.txt'.", file=sys.stderr)
        sys.exit(1)

    print(f"Targeting endpoint: {API_BASE_URL}/chat/completions\n")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    base_payload = {
        "messages": [{"role": "user", "content": "Health check."}],
        "stream": False,
        "max_tokens": 10,
    }
    url = f"{API_BASE_URL.rstrip('/')}/chat/completions"

    for model in MODELS_TO_TEST:
        print(f"--- Testing Model: {model} ---")
        payload = copy.deepcopy(base_payload)
        payload["model"] = model
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60,
                verify=False,  # Bypass SSL verification for self-signed certs
            )
            print(f"  Status Code: {response.status_code}")
            print(f"  Response: {response.text[:500]}")
        except requests.exceptions.RequestException as e:
            print(f"  Request failed: {e}")
        print("-" * (len(model) + 20) + "\n")


if __name__ == "__main__":
    key = get_api_key()
    run_test(key)
