import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# --- 1. CONFIGURAÇÃO ---
ARQUIVO_DATASET = os.path.join("dataset", "dataset_final_balanceado.csv")
MODELO_BERT = "neuralmind/bert-base-portuguese-cased"

# Nossas 6 etiquetas (IMPORTANTE: A ordem deve ser a mesma)
LABELS = ["COMPOSICAO", "INDICACAO", "CONTRAINDICACAO", "POSOLOGIA", "EFEITOS_ADVERSOS", "OUTROS"]

# Mapear de string (ex: "POSOLOGIA") para número (ex: 3)
label2id = {label: i for i, label in enumerate(LABELS)}
# Mapear de número (ex: 3) para string (ex: "POSOLOGIA")
id2label = {i: label for i, label in enumerate(LABELS)}

print(f"--- Carregando dataset: {ARQUIVO_DATASET} ---")
df = pd.read_csv(ARQUIVO_DATASET)

# O modelo não entende "POSOLOGIA", ele entende '3'. Vamos converter.
df['label'] = df['label'].map(label2id)

# Remover qualquer linha que não foi mapeada (segurança)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print("\n--- Dataset carregado e mapeado ---")
print(df.sample(5, random_state=42)) # Mostra 5 linhas aleatórias com o novo label numérico
print("\nNova distribuição (numérica):")
print(df['label'].value_counts().sort_index())


# --- 2. "MATRICULANDO" O BERTimbau ---
print(f"\n--- Baixando o modelo: {MODELO_BERT} ---")
print("(Isso pode demorar alguns minutos na primeira vez...)")

# 1. O "Dicionário" (Tokenizador)
# Ele sabe transformar "remédio" em números (ex: 8372)
tokenizer = AutoTokenizer.from_pretrained(MODELO_BERT)

# 2. O "Cérebro" (O Modelo)
# Carregamos o modelo e avisamos que ele precisa ter 6 saídas (labels)
model = AutoModelForSequenceClassification.from_pretrained(
    MODELO_BERT,
    num_labels=len(LABELS), # Nosso caso: 6
    id2label=id2label,
    label2id=label2id
)

print("\n--- SUCESSO! ---")
print("Modelo BERTimbau e Tokenizador carregados na memória.")
print(f"Pronto para a próxima etapa: Tokenizar e Treinar!")