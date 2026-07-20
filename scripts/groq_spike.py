"""One-off spike: confirm GROQ_API_KEY + chosen model + response_format=json_object
work against Groq's chat completions endpoint. Not part of the app — delete or keep
around for future debugging.

Usage:
    GROQ_API_KEY=... python scripts/groq_spike.py
"""

import json
import os
import sys

import requests

MODEL = "llama-3.3-70b-versatile"
URL = "https://api.groq.com/openai/v1/chat/completions"


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a energy-market analyst writing one short factual sentence. "
                    "Respond with strict JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    "The average DE-LU day-ahead electricity price over the last 7 days was "
                    "133.5 EUR/MWh, up from 93.3 EUR/MWh over the trailing year. "
                    'Respond with JSON: {"summary": "<one sentence>"}'
                ),
            },
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.4,
        "max_tokens": 200,
    }

    resp = requests.post(
        URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=20,
    )
    print("status:", resp.status_code)
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"]
    print("raw content:", content)
    parsed = json.loads(content)
    print("parsed:", parsed)


if __name__ == "__main__":
    main()
