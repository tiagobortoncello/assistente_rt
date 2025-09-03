# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import nltk
import json
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# O download de stopwords é necessário para a parte de resumo (se o modelo precisar) e é uma boa prática
nltk.download('stopwords', quiet=True)

# --- Configuração do Gemini API para geração de termos ---
# ATENÇÃO: Insira sua chave de API pessoal abaixo.
# Você pode obtê-la no Google AI Studio: https://aistudio.google.com/app/apikey
API_KEY = "AIzaSyBa9rAep3e1DO6SWWEPzdPAazjiHj6JzWc"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# --- Funções de Geração de Termos e Resumos ---

def gerar_termos_llm(texto):
    """
    Gera termos de indexação usando um LLM (Gemini Flash).
    O modelo é instruído a atuar como um indexador profissional e
    retornar os termos em formato JSON.
    """
    system_instruction = {
        "parts": [{
            "text": "Você é um indexador de documentos legislativos altamente experiente. Sua tarefa é analisar o texto de uma proposição e gerar uma lista de 5 a 8 termos de indexação que capturem os tópicos principais e o escopo do documento. Os termos devem ser curtos, precisos e refletir o conteúdo conceitual. Por exemplo: 'Comércio Eletrônico', 'Serviço de Atendimento ao Público'."
        }]
    }

    payload = {
        "contents": [{"parts": [{"text": texto}]}],
        "systemInstruction": system_instruction,
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {"type": "STRING"}
            }
        }
    }

    try:
        response = requests.post(
            f"{API_URL}?key={API_KEY}",
            headers={'Content-Type': 'application/json'},
            data=json.dumps(payload)
        )
        response.raise_for_status()
        result = response.json()
        
        # Extrai o texto gerado do formato da resposta
        json_string = result['candidates'][0]['content']['parts'][0]['text']
        termos_gerados = json.loads(json_string)
        
        # Formata a lista de termos
        return [t.strip() for t in termos_gerados if t]
    except requests.exceptions.RequestException as e:
        st.error(f"Erro na comunicação com a API: {e}")
        return []
    except (json.JSONDecodeError, KeyError) as e:
        st.error(f"Erro ao processar a resposta da API: {e}")
        return []

# Modelo de resumo em português (mantido do seu código original)
MODEL_NAME = "rhaymison/flan-t5-portuguese-small-summarization"

@st.cache_resource
def load_summarizer():
    """Carrega o modelo de resumo com cache para evitar recarregar."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    summarizer_pipeline = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=-1 # força CPU
    )
    return summarizer_pipeline

summarizer = load_summarizer()

def gerar_resumo(texto, tipo):
    """Gera um resumo do texto no estilo legislativo."""
    prompt = f"Resuma o seguinte {tipo} em estilo legislativo, texto corrido, objetivo, sem perder informações essenciais:\n\n{texto}"
    resumo = summarizer(prompt, max_length=120, min_length=40, do_sample=False)
    return resumo[0]["summary_text"]

# --- Interface Streamlit ---

st.set_page_config(page_title="Assistente de indexação e resumos", layout="wide")

st.title("Assistente de indexação e resumos")
st.subheader("Gerência de Informação Legislativa – GIL/GDI")
st.write("---")

tipo = st.selectbox("Escolha o tipo de documento:", ["Proposição", "Requerimento"])
texto_input = st.text_area("Cole a ementa ou texto aqui:", height=200)

if st.button("Gerar sugestões e resumo"):
    if texto_input.strip() == "":
        st.warning("Por favor, insira um texto para análise.")
    else:
        # Sugestão de termos usando o novo método com LLM
        st.markdown("### 🔑 Termos sugeridos")
        
        # Exibe um spinner enquanto a API está trabalhando
        with st.spinner('Gerando termos...'):
            termos_sugeridos = gerar_termos_llm(texto_input)
            st.write(", ".join(termos_sugeridos) if termos_sugeridos else "Nenhum termo sugerido.")

        # Resumo
        st.markdown("### 📝 Resumo gerado")
        with st.spinner('Gerando resumo...'):
            resumo = gerar_resumo(texto_input, tipo)
            st.success(resumo)
