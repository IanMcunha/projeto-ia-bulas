import pandas as pd
import os

# Nossa lista de exemplos [ (texto_da_bula, etiqueta_correta) ]
dados_etiquetados = []

# --- PROCESSO DE ETIQUETAGEM MANUAL (Bula da Dipirona) ---
# Aqui usamos os dados que extraímos do PDF

# 1. COMPOSICAO 
texto = "Cada mL da solução injetável contém: dipirona monoidratada.... 500 mg Excipiente: água para injetáveis."
dados_etiquetados.append( (texto, "COMPOSICAO") )

# 2. INDICAÇÃO 
texto = "Este medicamento é indicado como analgésico (para dor) e antitérmico (para febre)."
dados_etiquetados.append( (texto, "INDICACAO") )

# 3. CONTRAINDICAÇÃO 
texto = "A dipirona monoidratada não deve ser utilizada caso você tenha: - reações alérgicas, tais como reações cutâneas graves com este medicamento;"
dados_etiquetados.append( (texto, "CONTRAINDICACAO") )

texto = "Este medicamento é contraindicado para menores de 3 meses de idade ou pesando menos de 5 kg."
dados_etiquetados.append( (texto, "CONTRAINDICACAO") )

# 4. POSOLOGIA (Como usar) 
texto = "Adultos e adolescentes acima de 15 anos: em dose única de 2 a 5 mL (intravenosa e intramuscular); dose máxima diária de 10 mL."
dados_etiquetados.append( (texto, "POSOLOGIA") )

# 5. EFEITOS ADVERSOS 
texto = "A dipirona pode causar choque anafilático, reações anafiláticas/anafilactoides que podem se tornar graves com risco à vida e, em alguns casos, serem fatais."
dados_etiquetados.append( (texto, "EFEITOS_ADVERSOS") )

texto = "Reações hipotensivas isoladas. Podem ocorrer ocasionalmente após a administração, reações hipotensivas transitórias isoladas;"
dados_etiquetados.append( (texto, "EFEITOS_ADVERSOS") )

# 6. OUTROS (Qualquer coisa que não se encaixa) 
texto = "SANTISA LABORATÓRIO FARMACÊUTICO"
dados_etiquetados.append( (texto, "OUTROS") )

texto = "Solução injetável 500 mg/mL embalagem com 100 ampolas de 2 mL."
dados_etiquetados.append( (texto, "OUTROS") )

texto = "Siga a orientação de seu médico, respeitando sempre os horários, as doses e a duração do tratamento."
dados_etiquetados.append( (texto, "OUTROS") )

# --- Fim da etiquetagem ---

print(f"--- Total de exemplos etiquetados: {len(dados_etiquetados)} ---")

# Agora, vamos transformar isso em um DataFrame (tabela) do pandas
df = pd.DataFrame(dados_etiquetados, columns=['texto', 'label'])

# E salvar em um arquivo CSV!

# Cria a pasta 'dataset' se ela não existir
output_dir = "dataset"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Pasta '{output_dir}' criada com sucesso.")

caminho_csv = os.path.join(output_dir, "bula_dipirona_etiquetada.csv")
df.to_csv(caminho_csv, index=False)

print(f"Sucesso! Dataset salvo em: {caminho_csv}")
print("\nAmostra do dataset:")
print(df.head()) # Mostra as 5 primeiras linhas