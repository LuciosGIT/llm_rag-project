import requests
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.debug = True

HF_API_URL = os.getenv('HF_API_URL')

HF_API_TOKEN = os.getenv('HF_API_TOKEN')

if not HF_API_URL:
    logger.error("URL da API do Hugging Face não foi definida! Certifique-se de configurar a variável HF_API_URL.")
    raise ValueError("HF_API_URL não pode estar vazia.")

if not HF_API_TOKEN:
    logger.error("Token da API do Hugging Face não foi definida! Certifique-se de configurar a variável HF_API_TOKEN.")
    raise ValueError("HF_API_TOKEN não pode estar vazia.")


def query_hf_api(user_query, retries=2, delay=5):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    full_prompt =f"""
                    Você é um assistente especializado em fornecer respostas diretas e concisas, evitando redundâncias.
                    Seu objetivo é responder com precisão, usando informações relevantes e mantendo um tom profissional e neutro.
                    Seja claro e objetivo, sem introduções ou conclusões desnecessárias.

                    Pergunta do usuário: "{user_query}"
                    Resposta:
            """

    payload = {
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [
            {"role": "system", "content": full_prompt},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.2,
        "max_tokens": 200
    }

    for attempt in range(retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            logger.info(f"O prompt enviado pelo usuário foi: {full_prompt}")
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.warning(f"Tentativa {attempt}/{retries} falhou: {e}")
            time.sleep(delay)

    logger.error("Falha ao obter resposta da API após múltiplas tentativas.")
    return None


@app.route("/chat", methods=["POST"])
def chat():

    try: 
        data = request.get_json(silent=True) or {}
        user_query = data.get("query", "").strip()

        if not user_query:
            return jsonify({"error": "Campo 'query' é obrigatório"}), 400

    

        hf_response = query_hf_api(user_query)


        if hf_response is None:
            return jsonify({"error": "O campo 'query' é obrigatório"}), 500

        answer = hf_response["choices"][0]["message"]["content"].strip()
        return jsonify({"response": answer})
    
    except Exception as e:
        logger.exception("Error inesperado ao processar requisição.")
        return jsonify({"error": "Erro interno no servidor"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)
