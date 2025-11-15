import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# --- 1. CONFIGURAÇÃO ---
ARQUIVO_DATASET = os.path.join("dataset", "dataset_final_balanceado.csv")
MODELO_BERT = "neuralmind/bert-base-portuguese-cased"
RANDOM_STATE = 42

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
    label2id=label2id,
    use_safetensors=True
)

print("\n--- SUCESSO! ---")
print("Modelo BERTimbau e Tokenizador carregados na memória.")
print(f"Pronto para a próxima etapa: Tokenizar e Treinar!")

# --- FIM DO CARREGAMENTO ---
# O código abaixo cuida de todo o processo de Fine-Tuning:
# 1. (datasets): Converte e "tokeniza" nosso DataFrame.
# 2. (datasets): Divide os dados em conjuntos de Treino e Teste.
# 3. (scikit-learn): Define a função para calcular as métricas (nossa "régua").
# 4. (transformers): Configura o "Trainer" com os argumentos de treino.
# 5. (transformers): Executa o treino (trainer.train()).
# 6. (transformers): Salva o modelo treinado na pasta "modelo_bulario_bertimbau".
# ---

import numpy as np
from datasets import Dataset, DatasetDict
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

# --- 3. PREPARANDO OS DADOS PARA O TREINO ---

print("\n--- Convertendo DataFrame para Dataset Hugging Face ---")
# Converte o DataFrame do pandas para o formato que o 'transformers' gosta
dataset = Dataset.from_pandas(df)

print("--- Tokenizando o dataset (convertendo texto em números) ---")
# Esta função vai pegar o texto (ex: "Tome 2 comprimidos") e quebrar em "tokens"
def tokenizar_funcao(exemplos):
    # 'padding="max_length"' garante que todas as frases tenham o mesmo tamanho
    # 'truncation=True' corta frases que são longas demais
    return tokenizer(exemplos["texto"], padding="max_length", truncation=True, max_length=128)

# Aplica a função de tokenização em todo o dataset de uma vez (rápido!)
dataset_tokenizado = dataset.map(tokenizar_funcao, batched=True)

print("--- Divindo em Treino e Teste (80% para estudar, 20% para a prova) ---")
# 'train_test_split' divide nosso dataset. 80% treino, 20% teste
dataset_dividido = dataset_tokenizado.train_test_split(test_size=0.2, seed=RANDOM_STATE)

print("\nDataset pronto para o treino:")
print(dataset_dividido)


# --- 4. CONFIGURANDO A AVALIAÇÃO (A "RÉGUA") ---

def calcular_metricas(pred):
    """Função que o Trainer vai usar para calcular a precisão do modelo."""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1) # Pega a classe com maior probabilidade
    
    # Usando o scikit-learn que instalamos
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- 5. CONFIGURANDO O TREINADOR ---

# Define onde o modelo será salvo
DIRETORIO_SAIDA = "modelo_bulario_bertimbau"

# Se o seu PC for um pouco mais lento ou não tiver uma placa de vídeo (GPU) forte,
# mude o 'per_device_train_batch_size' de 16 para 8 ou 4.
TAMANHO_LOTE = 16 # Mude para 8 ou 4 se tiver problemas de memória

training_args = TrainingArguments(
    output_dir=DIRETORIO_SAIDA,          # Pasta para salvar o modelo
    num_train_epochs=3,                # Quantas vezes o modelo vai "ler" os dados (3 é um bom começo)
    per_device_train_batch_size=TAMANHO_LOTE,    # Quantos exemplos por vez (lotes)
    per_device_eval_batch_size=TAMANHO_LOTE,     # O mesmo para a avaliação
    weight_decay=0.01,                 # Regularização
    logging_dir='./logs',              # Pasta para logs
    logging_steps=10,                  # A cada 10 passos, mostra o progresso
    eval_strategy="epoch",             # No final de cada "leitura" (epoch), roda a "prova"
    save_strategy="epoch",             # Salva o modelo a cada epoch
    load_best_model_at_end=True,       # No final, recarrega o melhor modelo que ele encontrou
)

# Cria o "Gerente" do Treino
trainer = Trainer(
    model=model,                         # O cérebro do BERTimbau
    args=training_args,                  # As regras do treino
    train_dataset=dataset_dividido["train"], # O material de estudo
    eval_dataset=dataset_dividido["test"],   # A prova
    compute_metrics=calcular_metricas,     # A "régua" para dar a nota
    tokenizer=tokenizer,
)

# --- 6. TREINAR! ---
print("\n--- INICIANDO O TREINAMENTO! ---")
print("(Isso pode demorar alguns minutos...)")

trainer.train()

print("\n--- TREINAMENTO CONCLUÍDO! ---")

# --- 7. SALVAR O MODELO FINAL ---
print("\nSalvando o modelo final treinado...")
trainer.save_model(DIRETORIO_SAIDA)
tokenizer.save_pretrained(DIRETORIO_SAIDA)

print(f"\n--- SUCESSO! MODELO SALVO EM: {DIRETORIO_SAIDA} ---")

print("\n--- AVALIAÇÃO FINAL DO MODELO ---")
# Roda a avaliação final no set de teste
evaluation_results = trainer.evaluate()
print("Resultados da avaliação final:")
print(evaluation_results)