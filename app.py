import streamlit as st
import requests
import json
import os

# Função para carregar o dicionário de termos de um arquivo de texto
def carregar_dicionario_termos(nome_arquivo):
    """
    Carrega os termos de um arquivo de texto, construindo um mapa de hierarquia.
    Cada linha do arquivo representa um termo.
    Hierarquias são indicadas por '>' (ex: 'Termo Genérico > Termo Específico').
    """
    termos = []
    mapa_hierarquia = {}
    
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                partes = [p.strip() for p in line.split('>')]
                
                # Adiciona o termo mais específico (o último na cadeia)
                termo_especifico = partes[-1]
                if termo_especifico:
                    termos.append(termo_especifico)
                
                # Se houver hierarquia, mapeia a relação pai -> filho
                if len(partes) > 1:
                    termo_pai = partes[-2]
                    if termo_pai not in mapa_hierarquia:
                        mapa_hierarquia[termo_pai] = []
                    mapa_hierarquia[termo_pai].append(termo_especifico)
                    
    except FileNotFoundError:
        st.error(f"Erro: O arquivo '{nome_arquivo}' não foi encontrado.")
        return [], {}
        
    return termos, mapa_hierarquia

# Função para aplicar a lógica de hierarquia nos termos sugeridos pelo modelo de IA
def aplicar_logica_hierarquia(termos_sugeridos, mapa_hierarquia):
    """
    Filtra a lista de termos sugeridos para manter apenas os mais específicos.
    Se um termo genérico e um específico da mesma hierarquia forem sugeridos,
    o termo genérico é removido.
    """
    termos_finais = set(termos_sugeridos)
    
    for termo_gen in mapa_hierarquia:
        if termo_gen in termos_finais:
            for termo_especifico in mapa_hierarquia[termo_gen]:
                if termo_especifico in termos_finais:
                    termos_finais.discard(termo_gen)
                    break
                    
    return list(termos_finais)

# Função para gerar resumo usando a API do Google Gemini
def gerar_resumo(texto_original):
    """
    Gera um resumo a partir do texto original usando a API do Google Gemini.
    As regras para o resumo são fixas no prompt.
    """
    
    API_KEY = os.environ.get("API_KEY")
    if not API_KEY:
        st.error("Erro: A chave de API não foi encontrada. Por favor, configure a variável de ambiente 'API_KEY'.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
    
    regras_adicionais = """
    - Use linguagem formal e evite gírias.
    - Mantenha um tom objetivo e neutro.
    - Enfatize os pontos principais da proposição, como a obrigatoriedade, os detalhes do atendimento e as penalidades.
    - Use verbos na terceira pessoa do singular.
    """

    prompt_resumo = f"""
    Resuma a seguinte proposição legislativa de forma clara e concisa. 
    Siga as seguintes regras:
    {regras_adicionais}
    
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
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Erro na comunicação com a API: {http_err}")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        
    return "Não foi possível gerar o resumo. Verifique a chave de API e tente novamente."

# Função para gerar termos de indexação usando a API do Google Gemini
def gerar_termos_llm(texto_original, termos_dicionario):
    """
    Gera termos de indexação a partir do texto original, utilizando um dicionário de termos.
    """

    API_KEY = os.environ.get("API_KEY")
    if not API_KEY:
        st.error("Erro: A chave de API não foi encontrada. Por favor, configure a variável de ambiente 'API_KEY'.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"

    prompt_termos = f"""
    A partir do texto abaixo, selecione até 10 (dez) termos de indexação relevantes.
    Os termos de indexação devem ser selecionados EXCLUSIVAMENTE da seguinte lista:
    {", ".join(termos_dicionario)}
    Se nenhum termo da lista for aplicável, a resposta deve ser uma lista vazia.
    A resposta deve ser uma lista JSON de strings.
    
    Texto da Proposição: {texto_original}
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt_termos}]}],
        "tools": [{"google_search": {}}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "STRING"
                }
            }
        }
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extrai a string JSON e a converte para uma lista Python
        json_string = result.get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text", "[]")
        termos_sugeridos = json.loads(json_string)
        
        return termos_sugeridos
        
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Erro na comunicação com a API: {http_err}")
    except json.JSONDecodeError:
        st.error("Erro ao decodificar a resposta JSON do modelo.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        
    return []

# Configuração da página e UI
st.set_page_config(page_title="Gerador de Termos e Resumos de Proposições")
st.title("Gerador de Termos e Resumos de Proposições")
st.write("Insira o texto de uma proposição legislativa para gerar um resumo e termos de indexação.")

# Carregar o dicionário de termos
termo_dicionario, mapa_hierarquia = carregar_dicionario_termos("dicionario_termos.txt")

# Área de texto para entrada da proposição
texto_proposicao = st.text_area(
    "Cole o texto da proposição aqui:", 
    height=300,
    placeholder="Ex: 'A presente proposição dispõe sobre a criação de um programa de incentivo...'")

# Botão para gerar resumo e termos
if st.button("Gerar Resumo e Termos"):
    if not texto_proposicao:
        st.warning("Por favor, cole o texto da proposição para continuar.")
    else:
        with st.spinner('Gerando resumo e termos...'):
            # Gera o resumo
            resumo_gerado = gerar_resumo(texto_proposicao)
            
            # Gera os termos usando o modelo de IA
            termos_sugeridos_brutos = gerar_termos_llm(texto_proposicao, termo_dicionario)
            
            # Aplica a lógica de hierarquia para priorizar termos específicos
            termos_finais = aplicar_logica_hierarquia(termos_sugeridos_brutos, mapa_hierarquia)

        # Exibir os resultados
        if resumo_gerado:
            st.subheader("Resumo")
            st.write(resumo_gerado)
        
        st.subheader("Termos de Indexação")
        if termos_finais:
            termos_str = ", ".join(termos_finais)
            st.success(termos_str)
        else:
            st.warning("Nenhum termo relevante foi encontrado no dicionário.")
