import requests
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
app.debug = True

HF_API_URL = os.getenv('HF_API_URL')

HF_API_TOKEN = os.getenv('HF_API_TOKEN')


def query_hf_api(user_query, retries=2, delay=5):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.2,
        "max_tokens": 200
    }

    for attempt in range(retries):
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json()

        print(f"Tentativa {attempt+1}/{retries} falhou: {response.text}")
        time.sleep(delay)

    response.raise_for_status()


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Campo 'query' é obrigatório"}), 400

    hf_response = query_hf_api(user_query)

    answer = hf_response["choices"][0]["message"]["content"].strip()
    return jsonify({"response": answer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)
