import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
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
            termos.extend(str(df.iloc[idx]["termos"]).split("|"))  # Ajustado para barra vertical
    termos = list(dict.fromkeys([t.strip() for t in termos if isinstance(t, str)]))

    # 🔹 Filtrar apenas termos que aparecem no texto
    termos = [t for t in termos if t.lower() in novo_texto.lower()]

    return termos[:5]

# ================================
# Modelo de resumo em português
# ================================
MODEL_NAME = "ml6team/mt5-small-portuguese-finetuned-summarization"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

summarizer = pipeline(
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device=-1  # força CPU
)

def gerar_resumo(texto, tipo):
    # Prompt simulando estilo legislativo
    prompt = f"Resuma o seguinte {tipo} em estilo legislativo, texto corrido, objetivo, sem perder informações essenciais:\n\n{texto}"
    resumo = summarizer(prompt, max_length=120, min_length=40, do_sample=False)
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
