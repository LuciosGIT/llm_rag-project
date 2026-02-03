import requests
import time
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
import logging
import difflib

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.debug = True

HF_API_URL = os.getenv('HF_API_URL')

HF_API_TOKEN = os.getenv('HF_API_TOKEN')

LOCAL_KNOWLEDGE_DATABASE = "source.txt"

if not HF_API_URL:
    logger.error("URL da API do Hugging Face não foi definida! Certifique-se de configurar a variável HF_API_URL.")
    raise ValueError("HF_API_URL não pode estar vazia.")

if not HF_API_TOKEN:
    logger.error("Token da API do Hugging Face não foi definida! Certifique-se de configurar a variável HF_API_TOKEN.")
    raise ValueError("HF_API_TOKEN não pode estar vazia.")


def load_local_knowledge():
    """
    Carrega a base de conhecimento local utilizada como fonte para RAG.
    Returns:
    list[str]: Lista de blocos de conhecimento extraídos do arquivo.

    """
    if not os.path.exists(LOCAL_KNOWLEDGE_DATABASE):
        logger.warning("Arquivo de conhecimento local não encontrado. Continuando sem RAG")
        return []
    with open(LOCAL_KNOWLEDGE_DATABASE, "r", encoding="utf-8") as f:
        return f.read().split("\n\n") #Divide em blocos de conhecimento
    
knowledge_base = load_local_knowledge()

def retrieve_relevant_passage(query):
    """
    Recupera o trecho mais relevante da base de conhecimento local com base em similaridade textual.

    Args:
        query (str): Texto da consulta ou pergunta do usuário.

    Returns:
        str: Trecho de conhecimento mais relevante ou mensagem padrão
        caso a base de conhecimento esteja vazia.

    """
    if not knowledge_base:
        return "Nenhuma informação adicional disponível."
    
    best_match = max(knowledge_base, key = lambda passage: difflib.SequenceMatcher(None, query, passage).ratio())

    return best_match


def query_hf_api(user_query, retries=2, delay=5):
    """
    Envia uma consulta à API da Hugging Face utilizando um modelo LLM com
    suporte a recuperação de contexto (RAG) a partir de uma base local.

    Args:
        user_query (str): Pergunta ou consulta enviada pelo usuário.
        retries (int, optional): Número máximo de tentativas em caso de falha
            na requisição. Padrão é 2.
        delay (int, optional): Tempo de espera em segundos entre as tentativas.
            Padrão é 5.

    Returns:
        dict | None: Resposta JSON retornada pela API da Hugging Face ou
        None caso todas as tentativas falhem.
    """
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    relevant_context = retrieve_relevant_passage(user_query)

    full_prompt =f""""
                    [INSTRUÇÕES]
                    Você é um assistente especializado em fornecer respostas diretas e consisas, evitando redundâncias. Sua resposta deve se basear no [CONHECIMENTO RELEVANTE] se houver.

                    [CONHECIMENTO RELEVANTE]
                    "{relevant_context}"

                    [PERGUNTA DO USUÁRIO]
                    "{user_query}"

                    **Resposta:** [Sua resposta direta]
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
