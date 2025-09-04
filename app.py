import streamlit as st
import requests
import json
import os
import re

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
                
                partes = [p.strip() for p in line.split('>') if p.strip()]
                
                if not partes:
                    continue

                # Adiciona o termo mais específico (o último na cadeia)
                termo_especifico = partes[-1]
                if termo_especifico:
                    termo_especifico = termo_especifico.replace('\t', '')
                    termos.append(termo_especifico)
                
                # Se houver hierarquia, mapeia a relação pai -> filho
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
def gerar_resumo(api_key, texto_original):
    """
    Gera um resumo a partir do texto original usando a API do Google Gemini.
    As regras para o resumo são fixas no prompt.
    """
    
    if not api_key:
        st.error("Erro: A chave de API não foi configurada. Por favor, insira sua chave no campo acima.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    
    regras_adicionais = """
    - Use linguagem formal e evite gírias.
    - Mantenha um tom objetivo e neutro.
    - Enfatize os pontos principais da proposição, como a obrigatoriedade, os detalhes do atendimento e as penalidades.
    - Use verbos na terceira pessoa do singular.
    - Inicie o resumo diretamente com um verbo na terceira pessoa do singular, sem sujeito explícito.
    - Não inclua a parte sobre a vigência da lei (ex: "a lei entra em vigor na data de sua publicação").
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
def gerar_termos_llm(api_key, texto_original, termos_dicionario):
    """
    Gera termos de indexação a partir do texto original, utilizando um dicionário de termos.
    A resposta é esperada em formato de lista JSON.
    """
    if not api_key:
        st.error("Erro: A chave de API não foi configurada. Por favor, insira sua chave no campo acima.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    # Ajusta o prompt para solicitar uma lista JSON explicitamente, sem usar o responseSchema.
    prompt_termos = f"""
    A partir do texto abaixo, selecione até 10 (dez) termos de indexação relevantes.
    Os termos de indexação devem ser selecionados EXCLUSIVAMENTE da seguinte lista:
    {", ".join(termos_dicionario)}
    Se nenhum termo da lista for aplicável, a resposta deve ser uma lista JSON vazia: [].
    A resposta DEVE ser uma lista JSON de strings, sem texto adicional antes ou depois.
    
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
        
        # Extrai a string que o modelo retornou
        json_string = result.get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text", "[]")
        
        # Lógica de validação mais robusta
        termos_sugeridos = []
        try:
            # Tenta decodificar o JSON diretamente
            termos_sugeridos = json.loads(json_string)
            if not isinstance(termos_sugeridos, list):
                termos_sugeridos = []
        except json.JSONDecodeError:
            # Se a decodificação falhar, tenta extrair a lista manualmente
            try:
                # Procura a lista entre colchetes
                start = json_string.find('[')
                end = json_string.rfind(']')
                if start != -1 and end != -1:
                    json_string_limpa = json_string[start:end+1]
                    termos_sugeridos = json.loads(json_string_limpa)
                    if not isinstance(termos_sugeridos, list):
                        termos_sugeridos = []
                else:
                    termos_sugeridos = []
            except Exception:
                # Se tudo falhar, retorna uma lista vazia
                termos_sugeridos = []
        
        return termos_sugeridos
        
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Erro na comunicação com a API: {http_err}")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        
    return []

def processar_utilidade_publica(texto):
    """
    Verifica se o texto é uma declaração de utilidade pública e retorna termos e resumo
    formatados de acordo com a regra.
    """
    # Verifica se a frase 'Declara de utilidade pública' está no texto
    if "Declara de utilidade pública" in texto:
        # A regex foi atualizada para ser mais robusta, capturando o nome do município
        # seguido por uma vírgula, a palavra 'e', um ponto, uma nova linha ou o fim do texto.
        match = re.search(r"do Município de (.+?)(?:,| e |\.|\n|$)", texto, re.IGNORECASE)
        municipio = "Município não encontrado"
        if match:
            # Captura o nome do município
            municipio = match.group(1).strip()
            # Remove vírgula ou ponto no final do nome, caso a regex não o faça
            if municipio.endswith(','):
                municipio = municipio[:-1]
        
        resumo = "#"
        termos = ["Utilidade Pública", municipio]
        
        return resumo, termos
    
    return None, None

# Configuração da página e UI
st.set_page_config(page_title="Gerador de Termos e Resumos de Proposições")

# Título e Subtítulo Centralizados usando HTML
st.markdown("<h1 style='text-align: center;'>Gerador de Termos e Resumos de Proposições</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Gerência de Informação Legislativa – GIL/GDI</h3>", unsafe_allow_html=True)

st.write("Insira o texto de uma proposição legislativa para gerar um resumo e termos de indexação.")

# Campo para o usuário colar a chave da API
API_KEY = st.text_input("Cole sua chave de API aqui:", type="password")

# Dicionário de tipos de documento e seus respectivos arquivos de thesaurus
TIPOS_DOCUMENTO = {
    "Documentos Gerais": "dicionario_termos.txt"
}

# Caixa de seleção para tipo de documento
tipo_documento_selecionado = st.selectbox(
    "Selecione o tipo de documento:",
    options=list(TIPOS_DOCUMENTO.keys()),
)

# Carregar o dicionário de termos com base na seleção
arquivo_dicionario = TIPOS_DOCUMENTO[tipo_documento_selecionado]
termo_dicionario, mapa_hierarquia = carregar_dicionario_termos(arquivo_dicionario)

# Área de texto para entrada da proposição
texto_proposicao = st.text_area(
    "Cole o texto da proposição aqui:", 
    height=300,
    placeholder="Ex: 'A presente proposição dispõe sobre a criação de um programa de incentivo...'")

# Botão para gerar resumo e termos
if st.button("Gerar Resumo e Termos"):
    if not API_KEY:
        st.warning("Por favor, insira sua chave de API para continuar.")
    elif not texto_proposicao:
        st.warning("Por favor, cole o texto da proposição para continuar.")
    else:
        with st.spinner('Gerando resumo e termos...'):
            # Verifica a regra especial para utilidade pública
            resumo_gerado, termos_finais = processar_utilidade_publica(texto_proposicao)

            if resumo_gerado is None:
                # Se não for uma declaração de utilidade pública, segue o fluxo normal
                resumo_gerado = gerar_resumo(API_KEY, texto_proposicao)
                termos_sugeridos_brutos = gerar_termos_llm(API_KEY, texto_proposicao, termo_dicionario)
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
