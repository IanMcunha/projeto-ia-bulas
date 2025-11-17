import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# --- 1. CONFIGURAÇÃO ---
MODELO_SALVO = "modelo_bulario_bertimbau"
ACURACIA_MODELO = "95.1%"

LABELS = ["COMPOSICAO", "INDICACAO", "CONTRAINDICACAO", "POSOLOGIA", "EFEITOS_ADVERSOS", "OUTROS"]
id2label = {i: label for i, label in enumerate(LABELS)}

# Cores das tags de resultado
TAG_COLORS = {
    "COMPOSICAO": ("#007bff", "#ffffff"),
    "INDICACAO": ("#28a745", "#ffffff"),
    "CONTRAINDICACAO": ("#dc3545", "#ffffff"),
    "POSOLOGIA": ("#17a2b8", "#ffffff"),
    "EFEITOS_ADVERSOS": ("#ffc107", "#333333"),
    "OUTROS": ("#6c757d", "#ffffff"),
}

st.set_page_config(page_title="Classificador de Bulas", layout="wide")


# --- 2. CSS CUSTOMIZADO (Card opaco colorido + sidebar glass + texto escuro) ---
def carregar_css():
    css = """
    <style>
        :root {
            color-scheme: light;
        }

        * {
            box-sizing: border-box;
        }

        /* Texto base sempre escuro para evitar branco sobre branco */
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        .block-container,
        p, span, li, label {
            color: #111827;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #111827;
        }

        a {
            color: #2563eb;
        }

        /* Fundo gradiente geral */
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at top left, #f3e8ff 0, #e0f2fe 30%, #edf2f7 60%, #e5e7eb 100%);
        }

        .stApp {
            background: transparent;
        }

        /* Header transparente para o gradiente aparecer atrás */
        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            padding-top: 2rem;
        }

        /* Coluna 1: card principal opaco com cor suave, não branco puro */
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(1) > div {
            background: #f4f4fb;
            padding: 2.5rem 2.25rem;
            border-radius: 18px;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.15);
            border: 1px solid rgba(148, 163, 184, 0.35);
        }

        /* Coluna 2: card lateral com glassmorphism */
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) > div {
            margin-left: 0.5rem;
            background: rgba(255, 255, 255, 0.78);
            border-radius: 20px;
            padding: 1.75rem 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.90);
            box-shadow: 0 14px 40px rgba(15, 23, 42, 0.22);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
        }

        /* Título e subtítulo principais */
        .main-title {
            font-size: 1.9rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
            color: #020617;
        }

        .subtitle {
            font-size: 0.95rem;
            color: #4b5563;
            margin-bottom: 1.5rem;
        }

        /* Label do text area */
        .block-container label {
            font-weight: 600;
            color: #111827;
        }

        /* Text area acessível, fundo suave colorido e texto escuro */
        textarea {
            background-color: #edf2ff !important;
            color: #020617 !important;
            border-radius: 14px !important;
            border: 1px solid #c7d2fe !important;
            padding: 0.75rem 0.85rem !important;
            font-size: 0.95rem !important;
        }

        textarea::placeholder {
            color: #9ca3af !important;
        }

        textarea:focus {
            outline: none !important;
            border-color: #5e5df0 !important;
            box-shadow: 0 0 0 1px #5e5df0;
        }

        /* Botão principal em estilo pill e alto contraste */
        .stButton > button {
            background-color: #5e5df0;
            color: #ffffff;
            border-radius: 999px;
            border: none;
            padding: 0.65rem 1.8rem;
            font-weight: 600;
            font-size: 0.95rem;
            margin-top: 0.9rem;
            cursor: pointer;
            transition: background-color 0.15s ease, transform 0.05s ease, box-shadow 0.15s ease;
            box-shadow: 0 10px 25px rgba(94, 93, 240, 0.45);
        }

        .stButton > button:hover {
            background-color: #4b4ad8;
            transform: translateY(-1px);
            box-shadow: 0 14px 30px rgba(94, 93, 240, 0.55);
        }

        .stButton > button:active {
            transform: translateY(0);
            box-shadow: 0 6px 18px rgba(94, 93, 240, 0.40);
        }

        /* Card de resultado */
        .result-card {
            margin-top: 1.8rem;
            padding: 1.25rem 1.25rem 1rem;
            background-color: #f9fafb;
            border-radius: 14px;
            border: 1px solid #e5e7eb;
        }

        .result-tag {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.25rem 0.95rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }

        .result-confidence {
            font-size: 0.85rem;
            color: #4b5563;
            margin-bottom: 0.25rem;
        }

        .result-text {
            font-size: 0.9rem;
            color: #111827;
        }

        /* Título do card de exemplos sempre escuro */
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) h3 {
            color: #1a1a2e !important;
            margin-bottom: 0.75rem;
        }

        /* Mini cards de exemplo */
        .example-card {
            background-color: #ffffff;
            border-radius: 14px;
            padding: 0.75rem 0.85rem;
            margin-top: 0.6rem;
            border: 1px solid rgba(148, 163, 184, 0.45);
        }

        .example-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: #5e5df0;
            margin-bottom: 0.25rem;
        }

        .example-text {
            font-size: 0.85rem;
            color: #111827;
        }

        [data-testid="stNotification"] {
            border-radius: 12px;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# --- 3. CARREGAR O MODELO (Função com Cache) ---
@st.cache_resource
def carregar_modelo():
    if not os.path.exists(MODELO_SALVO):
        return None, None

    print("--- Carregando modelo e tokenizador do disco ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODELO_SALVO)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODELO_SALVO,
            use_safetensors=True
        )
        model.to("cpu")
        print("--- Modelo carregado com sucesso ---")
        return tokenizer, model
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None, None


tokenizer, model = carregar_modelo()
carregar_css()


# --- 4. INTERFACE DO STREAMLIT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        "<h1 class='main-title'>🤖 Classificador de Bulas (IA)</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p class='subtitle'>Este app usa um modelo BERTimbau open source, "
        f"treinado com <strong>{ACURACIA_MODELO} de acurácia</strong> para "
        f"classificar trechos de bulas da ANVISA em seis categorias.</p>",
        unsafe_allow_html=True,
    )

    if model is None or tokenizer is None:
        st.error(
            f"Erro crítico: a pasta do modelo treinado ('{MODELO_SALVO}') "
            "não foi encontrada. Verifique se o script de treino foi executado."
        )
    else:
        st.subheader("Cole um parágrafo de bula abaixo:")
        texto_usuario = st.text_area(
            "Texto da bula",
            height=200,
            placeholder='Exemplo: "Este medicamento é contraindicado para menores de 3 meses..."',
        )

        if st.button("Classificar Texto"):
            if texto_usuario.strip():
                print(f"Classificando o texto: {texto_usuario[:50]}...")

                inputs = tokenizer(
                    texto_usuario,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True,
                )
                inputs = {k: v.to("cpu") for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=-1)
                previsao_id = int(torch.argmax(probs).item())
                previsao_label = id2label[previsao_id]
                confidence = float(probs[previsao_id].item())

                bg_color, text_color = TAG_COLORS.get(
                    previsao_label, ("#6c757d", "#ffffff")
                )

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-tag"
                             style="background-color: {bg_color}; color: {text_color};">
                            {previsao_label}
                        </div>
                        <p class="result-confidence">
                            Confiança estimada: {confidence * 100:.1f}%.
                        </p>
                        <p class="result-text">
                            O modelo identificou que este trecho se parece mais com
                            a seção <strong>{previsao_label.replace("_", " ").title()}</strong> da bula.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Por favor, insira um texto para classificar.")


with col2:
    st.subheader("💡 Exemplos para Testar")

    exemplos = [
        {
            "titulo": "POSOLOGIA",
            "texto": (
                "Adultos e adolescentes acima de 15 anos: dose única de 2 a 5 mL "
                "por via intramuscular ou intravenosa; dose máxima diária de 10 mL."
            ),
        },
        {
            "titulo": "CONTRAINDICACAO",
            "texto": (
                "Este medicamento não pode ser usado por pessoas alérgicas à "
                "amoxicilina, a antibióticos penicilínicos ou cefalosporínicos."
            ),
        },
        {
            "titulo": "EFEITOS_ADVERSOS",
            "texto": (
                "Reações comuns: diarreia, náuseas e enjoo. Podem ocorrer ainda "
                "reações alérgicas cutâneas, como erupções e coceira."
            ),
        },
        {
            "titulo": "COMPOSICAO",
            "texto": (
                "Cada mL da solução injetável contém dipirona monoidratada 500 mg. "
                "Excipiente: água para injetáveis."
            ),
        },
        {
            "titulo": "INDICACAO",
            "texto": (
                "Este medicamento é indicado como analgésico e antitérmico, "
                "auxiliando no alívio da dor e da febre."
            ),
        },
        {
            "titulo": "OUTROS",
            "texto": (
                "Conservar o produto em temperatura ambiente, entre 15 °C e 30 °C, "
                "protegido da luz e da umidade."
            ),
        },
    ]

    for ex in exemplos:
        st.markdown(
            f"""
            <div class="example-card">
                <div class="example-title">{ex["titulo"]}</div>
                <p class="example-text">{ex["texto"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
