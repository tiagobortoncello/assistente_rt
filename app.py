# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import nltk
import json
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from collections import defaultdict
import os

# O download de stopwords é necessário para a parte de resumo (se o modelo precisar) e é uma boa prática
nltk.download('stopwords', quiet=True)

# --- Configuração do Gemini API para geração de termos ---
# ATENÇÃO: A chave de API agora é lida de uma variável de ambiente por segurança
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    st.error("Erro: A chave de API não foi encontrada. Por favor, configure a variável de ambiente 'API_KEY'.")
    st.stop()
    
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

# --- Funções de Leitura e Processamento do Dicionário ---
def carregar_dicionario_termos(filepath):
    """
    Carrega os termos de um arquivo de texto, com um termo por linha.
    Processa a hierarquia para criar uma lista plana e um mapa de relações.
    """
    termos_plano = []
    hierarquia_mapa = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                
                partes = [p.strip() for p in line.split('>')][::-1]
                
                # Adiciona o termo mais específico à lista plana
                termo_especifico = partes[0]
                termos_plano.append(termo_especifico)
                
                # Mapeia o termo específico ao seu genérico (se houver)
                if len(partes) > 1:
                    termo_generico = partes[1]
                    hierarquia_mapa[termo_especifico] = termo_generico
                    # Adiciona o termo genérico à lista plana se ainda não estiver lá
                    if termo_generico not in termos_plano:
                        termos_plano.append(termo_generico)
                        
    except FileNotFoundError:
        st.error(f"Erro: O arquivo '{filepath}' não foi encontrado. Por favor, crie o arquivo e adicione seus termos, um por linha.")
        return [], {}
        
    return termos_plano, hierarquia_mapa

# Carrega o dicionário de termos do arquivo e o mapa de hierarquia
TERMOS_DICIONARIO, HIERARQUIA_MAPA = carregar_dicionario_termos("dicionario_termos.txt")

def aplicar_logica_hierarquia(termos_sugeridos, hierarquia_mapa):
    """
    Filtra a lista de termos sugeridos para remover genéricos se um termo
    mais específico da mesma hierarquia estiver presente.
    """
    termos_sugeridos_set = {t.strip() for t in termos_sugeridos}
    termos_finais = set(termos_sugeridos_set)

    for termo in termos_sugeridos_set:
        termo_atual = termo
        while termo_atual in hierarquia_mapa:
            termo_generico = hierarquia_mapa[termo_atual]
            if termo_generico in termos_finais:
                termos_finais.discard(termo_generico)
            termo_atual = termo_generico
    
    return list(termos_finais)

# --- Funções de Geração de Termos e Resumos ---

def gerar_termos_llm(texto):
    """
    Gera termos de indexação usando um LLM (Gemini Flash), limitando
    as sugestões ao dicionário predefinido.
    """
    # Verifica se o dicionário está carregado
    if not TERMOS_DICIONARIO:
        return []

    system_instruction = {
        "parts": [{
            "text": f"""
                Você é um indexador de documentos legislativos altamente experiente.
                Sua tarefa é analisar o texto de uma proposição e gerar uma lista de 5 a 8 termos de indexação que capturem os tópicos principais e o escopo do documento.
                É absolutamente crucial que os termos sugeridos sejam escolhidos EXCLUSIVAMENTE da seguinte lista:
                {', '.join(TERMOS_DICIONARIO)}
                Se o texto não se encaixar em nenhum termo da lista, retorne uma lista vazia.
                """
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
        
        json_string = result['candidates'][0]['content']['parts'][0]['text']
        termos_gerados = json.loads(json_string)
        
        # Filtra os termos gerados para garantir que eles estão no dicionário
        dicionario_set = {t.strip().lower() for t in TERMOS_DICIONARIO}
        termos_filtrados = [
            t for t in termos_gerados if t.strip().lower() in dicionario_set
        ]

        # Aplica a lógica de hierarquia para remover genéricos desnecessários
        termos_finais = aplicar_logica_hierarquia(termos_filtrados, HIERARQUIA_MAPA)
        
        return [t.strip() for t in termos_finais if t]
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
    """Gera um resumo do texto no estilo legislativo com regras fixas."""
    regras_adicionais = "Use linguagem formal, evite gírias e mantenha um tom objetivo e neutro. O resumo deve usar verbos na terceira pessoa do singular."
    
    prompt_completo = f"Resuma o seguinte {tipo} em estilo legislativo, texto corrido, objetivo, sem perder informações essenciais:\n\n{texto}\n\nRegras adicionais: {regras_adicionais}"
    
    resumo = summarizer(prompt_completo, max_length=120, min_length=40, do_sample=False)
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
