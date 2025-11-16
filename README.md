# Projeto Semestral — Classificador de Bulas de Remédio (IA)

End-to-End AI Open Project — Ciência da Computação — Mackenzie — Turma 07N — 2025.2

## 👥 Grupo
- Arthur Vignati Moscardi — 10409688
- Enzo Bernal de Matos — 10402685
- Ian Miranda Da Cunha — 10409669
- Pedro Pessuto Rodrigues Ferreira — 10409729

---

## 🎯 Objetivo
Implementar um pipeline de "End-to-End AI" em Python para criar um classificador de texto. O objetivo é treinar um modelo de linguagem (LLM) 100% aberto para identificar e classificar automaticamente seções de bulas de remédio (ex: "Posologia", "Contraindicação", "Composição").

O projeto inclui as etapas de coleta de dados, etiquetagem automática, balanceamento de dataset, fine-tuning do modelo e, por fim, a criação de uma aplicação web (Streamlit) para consumir o modelo treinado.

---

## ⚙️ Pipeline do Projeto
O projeto é dividido em um pipeline de scripts Python que preparam os dados e treinam o modelo:

1.  **Coleta de Dados:** Os arquivos `.pdf` de bulas de remédio são baixados do portal [Bulário Eletrônico da ANVISA](https://www.gov.br/anvisa/pt-br/assuntos/medicamentos/bulas) e armazenados na pasta `/data`.

2.  **Etiquetagem Automática (`2_etiquetar_automatico.py`):**
    * O script lê todos os PDFs da pasta `/data` usando `PyMuPDF`.
    * Utiliza Expressões Regulares (Regex) para identificar os títulos das seções (ex: "6. COMO DEVO USAR...").
    * Segmenta o texto da bula em blocos e atribui uma etiqueta (label) a cada bloco de texto.
    * Salva um grande dataset não-balanceado (`dataset_completo_automatico.csv`).

3.  **Balanceamento (`3_balancear_dataset.py`):**
    * Analisa a distribuição de etiquetas e identifica um desbalanceamento (excesso de "OUTROS").
    * Aplica a técnica de **Undersampling**, mantendo 100% dos dados das classes importantes (ex: `POSOLOGIA`) e selecionando uma amostra aleatória de `OUTROS` de tamanho igual.
    * Salva o dataset final e balanceado (`dataset_final_balanceado.csv`).

4.  **Treinamento (`4_treinar_modelo.py`):**
    * Carrega o dataset balanceado.
    * Baixa o modelo LLM open-source **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) via Hugging Face.
    * Tokeniza o texto e divide os dados em conjuntos de treino (80%) e teste (20%).
    * Executa o *fine-tuning* do modelo usando PyTorch e a biblioteca `Trainer` (com aceleração de GPU/CUDA).

5.  **Resultado:**
    * O modelo treinado e o tokenizador são salvos na pasta `/modelo_bulario_bertimbau`, prontos para serem consumidos pela aplicação.

---

## 🧰 Tecnologias Utilizadas
- **Python 3.11 (64-bit)**: Linguagem principal do projeto.
- **PyTorch (`torch`):** O "motor" de deep learning para o treinamento via GPU.
- **Hugging Face `transformers`:** Para carregar o modelo BERTimbau e usar a API `Trainer` de fine-tuning.
- **Hugging Face `datasets`:** Para carregar e processar o dataset de forma eficiente.
- **Hugging Face `accelerate`:** Para otimizar o treino em diferentes hardwares (GPU/CPU).
- **Pandas:** Para manipulação inicial e análise dos arquivos `.csv`.
- **PyMuPDF (`fitz`):** Para a extração de texto de alta performance dos arquivos `.pdf`.
- **Scikit-learn (`sklearn`):** Para calcular as métricas de avaliação do modelo (Acurácia, F1-Score, etc.).
- **Streamlit:** (A ser implementado) Para a aplicação web de consumo do modelo.

---

## 🖥️ Ambiente e Execução
O projeto exige um ambiente Python 64-bit com suporte a CUDA (GPU NVIDIA) para o treinamento.

### 1. Pré-requisitos
- **Python 3.11 (64-bit)** (O projeto *não* é compatível com Python 3.12+ devido às dependências do PyTorch).
- **NVIDIA GPU** (ex: RTX 3050) com drivers CUDA 12.1 instalados.
- **Git** (para clonar o repositório).

### 2. Configuração do Ambiente
```bash
# 1. Clone o repositório
git clone <url_do_repositorio>
cd <pasta_do_projeto>

# 2. Crie o ambiente virtual (usando Python 3.11)
python -m venv .venv

# 3. Ative o ambiente
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 4. Instale o PyTorch (com suporte a GPU CUDA 12.1)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# 5. Instale o resto das dependências
pip install -r requirements.txt
```

### 3. Execução do Pipeline

Para gerar o dataset e treinar o modelo do zero, execute os scripts na seguinte ordem:

```bash
# 1. (Opcional) Adicione novos .pdf na pasta /data
# ...

# 2. Gera o dataset automático (dataset_completo_automatico.csv)
python 2_etiquetar_automatico.py

# 3. Balanceia o dataset (dataset_final_balanceado.csv)
python 3_balancear_dataset.py

# 4. Treina o modelo (salva em /modelo_bulario_bertimbau)
python 4_treinar_modelo.py
```

### 🏷️ Etiquetas de Classificação

O modelo é treinado para classificar um trecho de texto em uma das 6 categorias:

- `INDICACAO`  Para que o remédio serve.
- `COMPOSICAO`  Do que o remédio é feito.
- `CONTRAINDICACAO`  Quem não deve tomar.
- `POSOLOGIA`  Como e quanto tomar.
- `EFEITOS_ADVERSOS`  Quais males pode causar.
- `OUTROS`  Qualquer texto que não se encaixe nas anteriores (cabeçalhos, rodapés, seções de advertência, etc.).
