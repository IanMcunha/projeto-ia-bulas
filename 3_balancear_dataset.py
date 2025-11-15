import pandas as pd
import os

# --- 1. CONFIGURAÇÃO ---
ARQUIVO_ENTRADA = os.path.join("dataset", "dataset_completo_automatico.csv")
ARQUIVO_SAIDA = os.path.join("dataset", "dataset_final_balanceado.csv")

# Um "random_state" garante que a amostra aleatória seja sempre a mesma
RANDOM_STATE = 42

# --- 2. CARREGAR OS DADOS ---
print(f"Carregando dataset de: {ARQUIVO_ENTRADA}")
df = pd.read_csv(ARQUIVO_ENTRADA)

# --- 3. SEPARAR AS CLASSES ---
df_importante = df[df['label'] != 'OUTROS']
df_outros = df[df['label'] == 'OUTROS']

n_importante = len(df_importante)
n_outros = len(df_outros)

print(f"Encontrados {n_importante} exemplos importantes.")
print(f"Encontrados {n_outros} exemplos 'OUTROS'.")

# --- 4. BALANCEAMENTO (UNDERSAMPLING) ---
# Vamos pegar uma amostra aleatória de 'OUTROS' com o mesmo tamanho
# da nossa classe 'importante'
print(f"Balanceando: Pegando {n_importante} exemplos aleatórios de 'OUTROS'...")
df_outros_amostra = df_outros.sample(n=n_importante, random_state=RANDOM_STATE)

# --- 5. JUNTAR E EMBARALHAR ---
df_balanceado = pd.concat([df_importante, df_outros_amostra])

# Embaralha o dataset final (MUITO importante para o treino!)
df_balanceado = df_balanceado.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# --- 6. SALVAR ---
df_balanceado.to_csv(ARQUIVO_SAIDA, index=False)

print("\n--- SUCESSO! ---")
print(f"Dataset final balanceado salvo em: {ARQUIVO_SAIDA}")
print(f"Total de exemplos: {len(df_balanceado)}")
print("\nNova distribuição das etiquetas:")
print(df_balanceado['label'].value_counts())