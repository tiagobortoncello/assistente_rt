import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import nltk
from nltk.corpus import stopwords

# ================================
# Baixar stopwords em português
# ================================
nltk.download('stopwords')
stop_words_pt = stopwords.words('portuguese')

# ================================
# Carregamento do CSV fixo
# ================================
CSV_PATH = "proposicoes_treinamento.csv"  # Ajuste o caminho se necessário
df = pd.read_csv(CSV_PATH)

# ================================
# Preparação para sugestão de termos
# ================================
vectorizer = TfidfVectorizer(stop_words=stop_words_pt)
X = vectorizer.fit_transform(df["ementa"].fillna(""))

def sugerir_termos(novo_texto, top_n=5):
    vetor_novo = vectorizer.transform([novo_texto])
    similaridades = cosine_similarity(vetor_novo, X).flatten()
    indices = similaridades.argsort()[::-1][:top_n]
    termos = []
    for idx in indices:
        if pd.notna(df.iloc[idx]["termos"]):
            termos.extend(str(df.iloc[idx]["termos"]).split("|"))
    termos = list(dict.fromkeys([t.strip() for t in termos if isinstance(t, str)]))
    return termos[:5]

# ================================
# Modelo de resumo (gratuito)
# ================================
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

def gerar_resumo(texto, tipo):
    # Prompt simulando estilo legislativo do seu CSV
    prompt = f"Resuma o seguinte {tipo} em estilo legislativo, texto corrido, objetivo, sem perder informações essenciais:\n\n{texto}"
    resumo = summarizer(prompt, max_length=100, min_length=30, do_sample=False)
    return resumo[0]["summary_text"]

# ================================
# Interface Streamlit
# ================================
st.set_page_config(page_title="Assistente de indexação e resumos", layout="wide")

# Título e subtítulo
st.title("Assistente de indexação e resumos")
st.subheader("Gerência de Informação Legislativa – GIL/GDI")

st.write("---")

# Caixa cascata
tipo = st.selectbox("Escolha o tipo de documento:", ["Proposição", "Requerimento"])

# Entrada de texto
texto_input = st.text_area("Cole a ementa ou texto aqui:", height=200)

# Botão
if st.button("Gerar sugestões e resumo"):
    if texto_input.strip() == "":
        st.warning("Por favor, insira um texto para análise.")
    else:
        # Sugestão de termos
        termos_sugeridos = sugerir_termos(texto_input)
        st.markdown("### 🔑 Termos sugeridos")
        st.write(", ".join(termos_sugeridos) if termos_sugeridos else "Nenhum termo sugerido.")

        # Resumo
        st.markdown("### 📝 Resumo gerado")
        resumo = gerar_resumo(texto_input, tipo)
        st.success(resumo)
