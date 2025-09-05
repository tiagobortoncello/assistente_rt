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

# Função para obter a chave de API de forma segura
def get_api_key():
    """
    Tenta obter a chave de API de diferentes fontes:
    1. Streamlit secrets
    2. Variáveis de ambiente
    """
    # Tenta obter do st.secrets (método preferencial do Streamlit)
    api_key = st.secrets.get("GOOGLE_API_KEY")
    if api_key:
        return api_key
    
    # Se não encontrar, tenta obter das variáveis de ambiente
    api_key = os.environ.get("GOOGLE_API_KEY")
    return api_key

# Função para gerar resumo usando a API do Google Gemini
def gerar_resumo(texto_original):
    """
    Gera um resumo a partir do texto original usando a API do Google Gemini.
    As regras para o resumo são fixas no prompt.
    """
    api_key = get_api_key()
    
    if not api_key:
        st.error("Erro: A chave de API não foi configurada. Por favor, adicione-a como um segredo no GitHub ou em variáveis de ambiente.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"
    
    regras_adicionais = """
    - Mantenha o resumo em um único parágrafo, com no máximo 4 frases.
    - Use linguagem formal e evite gírias.
    - Mantenha um tom objetivo e neutro.
    - Use verbos na terceira pessoa do singular, na voz ativa.
    - Para descrever ações ou responsabilidades de autoridades, prefira o uso de verbos auxiliares como 'deve' ou 'pode' para indicar obrigação ou possibilidade, em vez de verbos no tempo presente do indicativo.
    - Evite o uso de verbos com partícula apassivadora ou de indeterminação do sujeito (Exemplo: "Aciona-se", "Cria-se").
    - Separe as siglas com o caractere "–". Por exemplo: 'MP – Medida Provisória'.
    - Inicie o resumo diretamente com um verbo na terceira pessoa do singular, sem sujeito explícito.
    - Evite iniciar frases subsequentes com 'Esta política', 'A lei' ou termos semelhantes. Prefira conectar as ideias de forma fluida ou iniciar a frase com um verbo, seguindo a regra inicial.
    - Não inclua a parte sobre a vigência da lei (ex: "a lei entra em vigor na data de sua publicação").
    - O resumo deve focar em três pontos principais:
        1. O que o programa institui e a quem se destina.
        2. Quem aciona o alerta e em que condições.
        3. Quais informações podem ser incluídas nas mensagens e quais tecnologias são permitidas, de forma geral, sem citar detalhes específicos ou exemplos.
    - O resumo não deve mencionar:
        - Detalhes sobre a Lei Geral de Proteção de Dados – LGPD.
        - Detalhes específicos sobre a Defesa Civil, ANATEL – Agência Nacional de Telecomunicações – ou outros órgãos.
        - Nomes específicos de programas (ex: "Alerta TEA-MG").
        - 'Minas Gerais' ou 'Estado de Minas Gerais'.
    """

    prompt_resumo = f"""
    Resuma a seguinte proposição legislativa de forma clara, concisa e com as regras abaixo.
    
    Regras para o Resumo:
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
def gerar_termos_llm(texto_original, termos_dicionario, num_termos):
    """
    Gera termos de indexação a partir do texto original, utilizando um dicionário de termos.
    A resposta é esperada em formato de lista JSON.
    """
    api_key = get_api_key()
    
    if not api_key:
        st.error("Erro: A chave de API não foi configurada. Por favor, adicione-a como um segredo no GitHub ou em variáveis de ambiente.")
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    # Ajusta o prompt para solicitar uma lista JSON explicitamente, sem usar o responseSchema.
    prompt_termos = f"""
    A partir do texto abaixo, selecione até {num_termos} termos de indexação relevantes.
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
        
        # Extrai a string que o modelo retornou, garantindo um valor padrão vazio.
        json_string = result.get("candidates", [])[0].get("content", {}).get("parts", [])[0].get("text", "")
        
        termos_sugeridos = []
        
        # Encontra todas as possíveis listas JSON na string
        matches = re.findall(r'(\[.*?\])', json_string, re.DOTALL)
        
        for match in matches:
            # Tenta limpar a string para um formato JSON válido, substituindo aspas simples por duplas.
            cleaned_string = match.replace("'", '"')
            
            # Tenta decodificar o trecho limpo
            try:
                parsed_list = json.loads(cleaned_string)
                
                # Valida se o resultado é uma lista e se todos os itens são strings
                if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                    termos_sugeridos = parsed_list
                    break # Encontrou uma lista válida, pode sair do loop
            except json.JSONDecodeError:
                continue # Se a decodificação falhar, continua para o próximo trecho
        
        return termos_sugeridos
        
    except requests.exceptions.HTTPError as http_err:
        st.error(f"Erro na comunicação com a API: {http_err}")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        
    return []

# Configuração da página e UI
st.set_page_config(page_title="Gerador de Termos e Resumos de Proposições")

# Título e Subtítulo Centralizados usando HTML
st.markdown("<h1 style='text-align: center;'>Gerador de Termos e Resumos de Proposições</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Gerência de Informação Legislativa – GIL/GDI</h3>", unsafe_allow_html=True)

st.write("Insira o texto de uma proposição legislativa para gerar um resumo e termos de indexação.")

# Dicionário de tipos de documento e seus respectivos arquivos de thesaurus
TIPOS_DOCUMENTO = {
    "Documentos Gerais": "dicionario_termos.txt"
}

# Modifica a caixa de seleção para incluir as opções "Proposição" e "Requerimento"
tipo_documento_selecionado = st.selectbox(
    "Selecione o tipo de documento:",
    options=["Proposição", "Requerimento"],
)

# Adiciona a caixa de seleção para o número de termos
num_termos_selecionado = st.selectbox(
    "Selecione a quantidade de termos de indexação:",
    options=["Até 3", "de 3 a 5", "5+"],
)

# Mapeia a seleção do usuário para um valor numérico para o prompt da IA
num_termos = 10 # Padrão para "5+"
if num_termos_selecionado == "Até 3":
    num_termos = 3
elif num_termos_selecionado == "de 3 a 5":
    num_termos = 5

# Carregar o dicionário de termos com base na seleção
arquivo_dicionario = TIPOS_DOCUMENTO["Documentos Gerais"] # Mantém o dicionário de termos fixo
termo_dicionario, mapa_hierarquia = carregar_dicionario_termos(arquivo_dicionario)

# Remove o termo "Minas Gerais (MG)" do dicionário para evitar sua sugestão
if "Minas Gerais (MG)" in termo_dicionario:
    termo_dicionario.remove("Minas Gerais (MG)")

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
            resumo_gerado = ""
            termos_finais = []
            
            # Regra específica para "utilidade pública" para fins de "servidão"
            match_servidao = re.search(r"declara de utilidade pública,.*servidão.*no Município de ([\w\s-]+)", texto_proposicao, re.IGNORECASE | re.DOTALL)
            
            # Regra genérica para "utilidade pública"
            match_utilidade_publica = re.search(r"declara de utilidade pública.*no Município de ([\w\s-]+)", texto_proposicao, re.IGNORECASE | re.DOTALL)
            
            if match_servidao:
                municipio = match_servidao.group(1).strip()
                termos_finais = ["Servidão Administrativa", municipio]
                resumo_gerado = "Não precisa de resumo."
            elif match_utilidade_publica:
                municipio = match_utilidade_publica.group(1).strip()
                termos_finais = ["Utilidade Pública", municipio]
                resumo_gerado = "Não precisa de resumo."
            else:
                # Lógica normal para as demais proposições
                if tipo_documento_selecionado == "Proposição":
                    resumo_gerado = gerar_resumo(texto_proposicao)
                elif tipo_documento_selecionado == "Requerimento":
                    resumo_gerado = "Não precisa de resumo."

                termos_sugeridos_brutos = gerar_termos_llm(texto_proposicao, termo_dicionario, num_termos)
                
                # Regra para adicionar "Política Pública" automaticamente
                if re.search(r"institui (?:a|o) (?:política|programa) estadual|cria (?:a|o) (?:política|programa) estadual", texto_proposicao, re.IGNORECASE):
                    if termos_sugeridos_brutos is not None and "Política Pública" not in termos_sugeridos_brutos:
                        termos_sugeridos_brutos.append("Política Pública")

                # Aplica a lógica de hierarquia para priorizar termos específicos
                if termos_sugeridos_brutos is not None:
                    termos_finais = aplicar_logica_hierarquia(termos_sugeridos_brutos, mapa_hierarquia)
                else:
                    termos_finais = []

        # Exibir os resultados
        st.subheader("Resumo")
        st.write(resumo_gerado)
        
        st.subheader("Termos de Indexação")
        if termos_finais:
            termos_str = ", ".join(termos_finais)
            st.success(termos_str)
        else:
            st.warning("Nenhum termo relevante foi encontrado no dicionário.")
