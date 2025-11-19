# Projeto Semestral: Classificador de Bulas de Remédio (IA)

End to End AI Open Project em Python  
Curso de Ciência da Computação - Mackenzie - Turma 07N - 2025.2

## 👥 Grupo

- Arthur Vignati Moscardi - 10409688  
- Enzo Bernal de Matos - 10402685  
- Ian Miranda Da Cunha - 10409669  
- Pedro Pessuto Rodrigues Ferreira - 10409729

##  Vídeo

https://www.youtube.com/watch?v=bqggGWBvkuY

---

## 1. Resumo do Projeto

Este projeto implementa um pipeline completo de inteligência artificial de ponta a ponta para classificar seções de bulas de remédio.

O objetivo foi desenvolver uma solução 100% open source capaz de:

1. Ler bulas da ANVISA em PDF.  
2. Converter o conteúdo em um dataset rotulado.  
3. Balancear as classes de forma controlada.  
4. Fazer fine tuning de um modelo de linguagem em português (BERTimbau).  
5. Disponibilizar um classificador em tempo real através de uma aplicação web em Streamlit.

O modelo final alcançou **95,1% de acurácia** no conjunto de teste, classificando parágrafos em seis categorias:

- `INDICACAO`  
- `COMPOSICAO`  
- `CONTRAINDICACAO`  
- `POSOLOGIA`  
- `EFEITOS_ADVERSOS`  
- `OUTROS`

---

## 2. Objetivo Detalhado

Implementar, em Python, um pipeline de End to End AI para classificar trechos de bulas de remédio em português, usando um modelo de linguagem 100% aberto.

O pipeline inclui:

- Coleta de dados em PDF a partir do Bulário Eletrônico da ANVISA.  
- Extração e etiquetagem automática de trechos usando Regex.  
- Balanceamento de classes para evitar viés em `OUTROS`.  
- Fine tuning do BERTimbau.  
- Exposição do modelo em uma aplicação web acessível (Streamlit).

---

## 3. Pipeline do Projeto

O projeto é dividido em um pipeline de scripts Python que preparam os dados e treinam o modelo.

### 3.1 Coleta de Dados

Os arquivos `.pdf` de bulas de remédio são baixados do portal oficial:

> Bulário Eletrônico da ANVISA  
> https://www.gov.br/anvisa/pt-br/assuntos/medicamentos/bulas

Esses arquivos são armazenados na pasta:

```text
/data
```

Foram utilizados 78 PDFs de bulas de diferentes medicamentos (por exemplo, Dipirona, Amoxicilina, Losartana).

### 3.2 Etiquetagem Automática (`2_etiquetar_automatico.py`)

Responsável por transformar PDFs não estruturados em um dataset rotulado.

- Lê todos os PDFs da pasta `/data` usando **PyMuPDF (`fitz`)**.  
- Extrai o texto de cada bula.  
- Utiliza **Expressões Regulares (Regex)** para identificar títulos de seções padronizadas da ANVISA  
  (por exemplo, "6. COMO DEVO USAR ESTE MEDICAMENTO").  
- Segmenta o texto em blocos e atribui uma etiqueta (label) a cada bloco.  
- Gera um dataset inicial não balanceado:

```text
dataset_completo_automatico.csv
```

Esse dataset contém aproximadamente 1453 exemplos.

### 3.3 Balanceamento (`3_balancear_dataset.py`)

Ao analisar o dataset bruto, foi identificado um forte desbalanceamento, com grande predominância de `OUTROS`.

Distribuição aproximada:

- `OUTROS`: 893 amostras (cerca de 61,5%)  
- Demais classes (5 classes importantes): 560 amostras (cerca de 38,5%)

Treinar diretamente nesse cenário geraria um modelo enviesado, tendendo a prever `OUTROS` para manter acurácia artificialmente alta.

Para corrigir isso, foi aplicada a técnica de **undersampling**:

- Mantidas 100% das amostras das classes importantes (por exemplo, `POSOLOGIA`, `INDICACAO`, etc.).  
- Selecionada uma amostra aleatória de `OUTROS` do mesmo tamanho das demais classes combinadas.

Resultado:

```text
dataset_final_balanceado.csv   # 1120 exemplos (560 classes importantes + 560 OUTROS)
```

Esse é o dataset final usado no treino.

### 3.4 Treinamento (`4_treinar_modelo.py`)

Script responsável pelo fine tuning do modelo.

- Carrega o dataset balanceado (`dataset_final_balanceado.csv`).  
- Baixa o modelo LLM open source **BERTimbau** (`neuralmind/bert-base-portuguese-cased`) via Hugging Face.  
- Tokeniza o texto e divide os dados em:
  - 80% treino (896 exemplos)  
  - 20% teste (224 exemplos)
- Executa o fine tuning usando:
  - `transformers` (API `Trainer`)  
  - PyTorch com aceleração GPU (CUDA)  
- Ao final, o modelo treinado e o tokenizador são salvos em:

```text
/modelo_bulario_bertimbau
```

Essa pasta é consumida diretamente pela aplicação web.

### 3.5 Aplicação Web (`app.py`)

Por fim, foi desenvolvida uma aplicação em **Streamlit** para consumo do modelo:

- Carrega o modelo e o tokenizador da pasta `/modelo_bulario_bertimbau`.  
- Expõe uma interface em que o usuário cola um parágrafo de bula.  
- Ao clicar em “Classificar Texto”, o app:
  - Tokeniza o texto.  
  - Passa o batch pelo BERTimbau fine tunado.  
  - Retorna a classe prevista.

Do ponto de vista de UX:

- A interface é organizada em duas colunas.  
- A coluna principal contém:
  - Título do app.  
  - Texto explicativo com a acurácia.  
  - Caixa de texto para entrada.  
  - Botão de classificação.  
  - Card de resultado com etiqueta colorida para a classe prevista.  
- A coluna lateral contém um card com efeito de glassmorphism e alguns exemplos prontos para teste.

---

## 4. Tecnologias Utilizadas

- **Python 3.11 (64 bits)**  
  Linguagem principal do projeto.

- **PyTorch (`torch`)**  
  Motor de deep learning para treino usando GPU.

- **Hugging Face `transformers`**  
  Para carregar o BERTimbau, definir o modelo de classificação e usar a API `Trainer`.

- **Hugging Face `datasets`**  
  Para manipulação e divisão do dataset de forma eficiente.

- **Hugging Face `accelerate`**  
  Para otimizar o treino em diferentes hardwares (GPU e CPU).

- **Pandas**  
  Para manipulação e análise dos arquivos `.csv`.

- **PyMuPDF (`fitz`)**  
  Para extração de texto de alta performance a partir de PDFs.

- **Scikit-learn (`sklearn`)**  
  Para cálculo de métricas de avaliação (acurácia, F1 score, precisão, etc.).

- **Streamlit**  
  Para a aplicação web de consumo do modelo.

---

## 5. Ambiente e Execução

O projeto foi pensado para treinar em GPU, mas a inferência via Streamlit roda tranquilamente em CPU.

### 5.1 Pré-requisitos

- **Python 3.11 (64 bits)**  
  O projeto não é compatível com Python 3.12+ devido às dependências atuais do PyTorch.

- **NVIDIA GPU** (por exemplo, RTX 3050) com drivers CUDA 12.1 instalados para o treino.

- **Git** para clonar o repositório.

### 5.2 Configuração do Ambiente

```bash
# 1. Clone o repositório
git clone <url_do_repositorio>
cd <pasta_do_projeto>

# 2. Crie o ambiente virtual (usando Python 3.11)
python -m venv .venv

# 3. Ative o ambiente
# Windows
.\.venv\Scripts ctivate
# macOS/Linux
# source .venv/bin/activate

# 4. Instale o PyTorch (com suporte a GPU CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Instale o resto das dependências
pip install -r requirements.txt
```

### 5.3 Execução do Pipeline

Para gerar o dataset e treinar o modelo do zero, execute os scripts na seguinte ordem:

```bash
# 1. (Opcional) Adicione novos PDFs na pasta /data
# ...

# 2. Gera o dataset automático (dataset_completo_automatico.csv)
python 2_etiquetar_automatico.py

# 3. Balanceia o dataset (dataset_final_balanceado.csv)
python 3_balancear_dataset.py

# 4. Treina o modelo (salva em /modelo_bulario_bertimbau)
python 4_treinar_modelo.py
```

---

## 6. Executando a Aplicação Web (Streamlit)

Depois de treinar o modelo ou copiar uma versão já treinada para a pasta `modelo_bulario_bertimbau`, basta rodar:

```bash
streamlit run app.py
```

Por padrão, o app fica disponível apenas na máquina local.

---

## 7. Etiquetas de Classificação

O modelo é treinado para classificar cada trecho de texto em uma das seis categorias abaixo:

- `INDICACAO`  
  Para que o remédio serve.

- `COMPOSICAO`  
  Do que o remédio é feito.

- `CONTRAINDICACAO`  
  Quem não deve tomar.

- `POSOLOGIA`  
  Como e quanto tomar.

- `EFEITOS_ADVERSOS`  
  Quais males pode causar.

- `OUTROS`  
  Qualquer texto que não se encaixe nas anteriores  
  (cabeçalhos, rodapés, seções de advertência, informações do fabricante, etc.).

---

## 8. Resultados e Conclusão

O crescimento do dataset (de um conjunto inicial pequeno para 78 bulas) e o uso de balanceamento via undersampling foram decisivos para a qualidade do modelo.

No conjunto de teste, com 224 exemplos nunca vistos durante o treino, o classificador atingiu:

- **Acurácia**: 95,1%  
- **F1 score (ponderado)**: 0,9517  
- **Precisão (ponderada)**: 0,9546  

Esses resultados mostram que:

- A estratégia de etiquetagem automática baseada em Regex foi suficientemente robusta para gerar dados de treino de boa qualidade.  
- O BERTimbau se mostrou adequado para o idioma e para o tamanho do problema.  
- O pipeline completo, da coleta ao app web, comprova a viabilidade de uma solução de classificação de bulas totalmente aberta, reproduzível e extensível para trabalhos futuros.
