import fitz  # PyMuPDF
import os

# O nome do arquivo que você salvou na pasta 'data'
nome_do_arquivo = "bula_dipirona.pdf" 

# Isso cria o caminho completo: "data/bula_dipirona.pdf"
arquivo_pdf = os.path.join("data", nome_do_arquivo) 

print(f"--- Tentando ler o arquivo: {arquivo_pdf} ---")

texto_completo = ""

try:
    with fitz.open(arquivo_pdf) as doc:
        print(f"Sucesso! O PDF tem {len(doc)} páginas.")
        for pagina_num, pagina in enumerate(doc):
            texto_completo += pagina.get_text()
            
    print("\n--- Texto extraído com sucesso! (Início da bula) ---")
    # Mostra os primeiros 1500 caracteres
    print(texto_completo[:1500])

except Exception as e:
    print(f"\n--- ERRO ---")
    print(f"Não consegui ler o arquivo. Erro: {e}")
    print("Verifique se:")
    print(f"1. O nome '{nome_do_arquivo}' está escrito exatamente igual ao arquivo.")
    print(f"2. O arquivo está mesmo dentro da pasta 'data'.")   