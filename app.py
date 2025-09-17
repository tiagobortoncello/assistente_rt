import streamlit as st
import pandas as pd
import requests
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Funções de dicionário e hierarquia ---
def carregar_dicionario_termos(nome_arquivo):
    termos = []
    mapa_hierarquia = {}
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                partes = [p.strip() for p in line.split('>') if p.strip()]
                if not partes:
                    continue
                termo_especifico = partes[-1].replace('\t', '')
                termos.append(termo_especifico)
                if len(partes) > 1:
                    termo_pai = partes[-2].replace('\t', '')
                    if termo_pai not in mapa_hierarquia:
                        mapa_hierarquia[termo_pai] = []
                    mapa_hierarquia[termo_pai].append(termo_especifico)
    except FileNotFoundError:
        st.error(f"Erro: O arquivo '{nome_arquivo}' não foi encontrado.")
        return [], {}
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o dicionário de termos: {e}")
        return [], {}
    return termos, mapa_hierarquia

def aplicar_logica_hierarquia(termos_sugeridos, mapa_hierarquia):
    termos_finais = set(termos_sugeridos)
    mapa_inverso_hierarquia = {}
    for pai, filhos in mapa_hierarquia.items():
        for filho in filhos:
            mapa_inverso_hierarquia[filho] = pai
    termos_a_remover = set()
    for termo in termos_sugeridos:
        if termo in mapa_inverso_hierarquia:
            termo_pai = mapa_inverso_hierarquia[termo]
            if termo_pai in termos_finais:
                termos_a_remover.add(termo_pai)
    termos_finais = termos_finais - termos_a_remover
    return list(termos_finais)

# --- Funções para API ---
def get_api_key():
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if api_key:
        return api_key
    api_key = os.environ.get("GOOGLE_API_KEY")
    return api_key

def gerar_resumo(texto_original, referencia_resumos=None):
    api_key = get_api_key()
    if not api_key:
        st.error("Erro: A chave de API não foi configurada.")
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    
    regras_adicionais = """
    - Mantenha o resumo em um único parágrafo, com no máximo 4 frases.
    - Use linguagem formal e neutra.
    - Use verbos na terceira pessoa do singular, voz ativa.
    - Não inclua parte sobre vigência da lei.
    - Inspire-se nos exemplos de resumo fornecidos.
    """
    
    prompt_resumo = f"""
    Resuma a proposição abaixo seguindo as regras e os exemplos:

    Regras:
    {regras_adicionais}

    Exemplos de Resumo: {referencia_resumos}

    Texto da Proposição: {texto_original}
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt_resumo}]}],
        "tools": [{"google_search": {}}]
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text", "")
    except Exception as e:
        st.error(f"Erro ao gerar resumo: {e}")
        return "Não foi possível gerar o resumo."

def gerar_termos_llm(texto_original, termos_referencia, num_termos):
    api_key = get_api_key()
    if not api_key:
        st.error("Erro: A chave de API não foi configurada.")
        return []
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    prompt_termos = f"""
    Selecione até {num_termos} termos de indexação do seguinte texto.
    Use como referência os termos: {termos_referencia}.
    Retorne apenas uma lista JSON de termos aplicáveis, sem texto extra.

    Texto da Proposição: {texto_original}
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt_termos}]}],
        "tools": [{"google_search": {}}]
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        json_string = result.get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text", "")
        matches = re.findall(r'(\[.*?\])', json_string, re.DOTALL)
        termos_sugeridos = []
        for match in matches:
            cleaned_string = match.replace("'", '"')
            try:
                parsed_list = json.loads(cleaned_string)
                if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                    termos_sugeridos = parsed_list
                    break
            except json.JSONDecodeError:
                continue
        return termos_sugeridos
    except Exception as e:
        st.error(f"Erro ao gerar termos: {e}")
        return []

# --- Funções para CSV e similaridade ---
def carregar_csv_modelo(nome_csv):
    try:
        df = pd.read_csv(nome_csv, encoding='utf-8')
        # Converte termos de string separada por | para lista
        df['termos'] = df['termos'].fillna('').apply(lambda x: [t.strip() for t in str(x).split('|') if t.strip()])
        df['Resumo'] = df['Resumo'].fillna('')
        return df
    except Exception as e:
        st.error(f"Erro ao carregar CSV: {e}")
        return pd.DataFrame()

def buscar_similares(texto_usuario, df_modelo, top_n=3):
    # Usando TF-IDF na combinação ementa + texto
    corpus = (df_modelo['ementa'].fillna('') + ' ' + df_modelo['texto'].fillna('')).tolist()
    tfidf = TfidfVectorizer(stop_words='portuguese')
    tfidf_matrix = tfidf.fit_transform(corpus + [texto_usuario])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    indices = cosine_sim[0].argsort()[::-1][:top_n]
    similares = df_modelo.iloc[indices]
    return similares

# --- Streamlit UI ---
st.set_page_config(page_title="Gerador de Termos e Resumos de Proposições")
st.markdown("<h1 style='text-align: center;'>Gerador de Termos e Resumos de Proposições</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Gerência de Informação Legislativa – GIL/GDI</h3>", unsafe_allow_html=True)

st.write("Insira o texto de uma proposição legislativa para gerar um resumo e termos de indexação.")

TIPOS_DOCUMENTO = {"Documentos Gerais": "dicionario_termos.txt"}
tipo_documento_selecionado = st.selectbox("Selecione o tipo de documento:", options=["Proposição", "Requerimento"])
num_termos_selecionado = st.selectbox("Selecione a quantidade de termos de indexação:", options=["Até 3", "de 3 a 5", "5+"])
num_termos = 10
if num_termos_selecionado == "Até 3":
    num_termos = 3
elif num_termos_selecionado == "de 3 a 5":
    num_termos = 5

# Carrega dicionário e CSV
arquivo_dicionario = TIPOS_DOCUMENTO["Documentos Gerais"]
termo_dicionario, mapa_hierarquia = carregar_dicionario_termos(arquivo_dicionario)
if "Minas Gerais (MG)" in termo_dicionario:
    termo_dicionario.remove("Minas Gerais (MG)")
df_modelo = carregar_csv_modelo("proposicoes_treinamento.csv")

texto_proposicao = st.text_area("Cole o texto da proposição aqui:", height=300, placeholder="Ex: 'A presente proposição dispõe sobre a criação de um programa de incentivo...'")

if st.button("Gerar Resumo e Termos"):
    if not texto_proposicao:
        st.warning("Por favor, cole o texto da proposição para continuar.")
    else:
        with st.spinner('Gerando resumo e termos...'):
            # 1. Buscar exemplos similares no CSV
            similares = buscar_similares(texto_proposicao, df_modelo, top_n=3)
            termos_referencia = []
            referencia_resumos = []
            for _, row in similares.iterrows():
                termos_referencia.extend(row['termos'])
                if row['Resumo']:
                    referencia_resumos.append(row['Resumo'])

            termos_referencia = list(set(termos_referencia))

            # 2. Gerar termos via IA usando referência do CSV
            if tipo_documento_selecionado == "Proposição":
                termos_sugeridos_brutos = gerar_termos_llm(texto_proposicao, termos_referencia, num_termos)
                resumo_gerado = gerar_resumo(texto_proposicao, referencia_resumos=referencia_resumos)
            else:
                termos_sugeridos_brutos = gerar_termos_llm(texto_proposicao, termos_referencia, num_termos)
                resumo_gerado = "Não precisa de resumo."

            # 3. Aplicar lógica de hierarquia
            if termos_sugeridos_brutos:
                termos_finais = aplicar_logica_hierarquia(termos_sugeridos_brutos, mapa_hierarquia)
            else:
                termos_finais = []

        st.subheader("Resumo")
        st.markdown(f"<p style='text-align: justify;'>{resumo_gerado}</p>", unsafe_allow_html=True)
        st.subheader("Termos de Indexação")
        if termos_finais:
            st.success(", ".join(termos_finais))
        else:
            st.warning("Nenhum termo relevante foi encontrado.")
