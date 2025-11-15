import os
import re
import pandas as pd
import fitz  # PyMuPDF

# --- 1. FUNÇÃO DE EXTRAÇÃO DE TEXTO ---
def extrair_texto_pdf(caminho_pdf: str) -> str:
    """Extrai o texto completo de um arquivo PDF."""
    texto = ""
    try:
        with fitz.open(caminho_pdf) as doc:
            for pagina in doc:
                texto += pagina.get_text()
        print(f"  -> Lido com sucesso.")
        return texto
    except Exception as e:
        print(f"  [AVISO] Falha ao ler {caminho_pdf}: {e}")
        return "" # Retorna vazio se o PDF for ilegível (ex: imagem)

# --- 2. O CÉREBRO: FUNÇÃO DE ETIQUETAGEM AUTOMÁTICA ---
def segmentar_e_etiquetar(texto: str):
    """Segmenta o texto por seções e etiqueta automaticamente."""
    
    # Lista de Regex para filtrar lixo (cabeçalhos/rodapés comuns)
    # Adicione mais aqui se vir mais padrões
    JUNK_REGEX = re.compile(
        r"Bula (para|do) paciente|eurofarma|CIMED|SANTISA|TEUTO|RANBAXY|GERMED|VITAMEDIC|MULTILAB|"
        r"Modelo de bula|Informações ao Paciente|IDENTIFICAÇÃO DO MEDICAMENTO|"
        r"Farm\. Resp|CNPJ|Indústria Brasileira|^\s*página \d+|\s*VP REV \d+",
        re.IGNORECASE
    )
    
    # Padrões de cabeçalhos mapeados para nossos rótulos
    # Adicionamos os números (ex: 1., 3.) para precisão
    # Adicionamos "STOP_LABEL" para voltar para "OUTROS"
    STOP_LABEL = "STOP_LABEL" 
    padroes = [
        (r"^(1|I)\.?\s*PARA QUE ESTE MEDICAMENTO( É| E)? INDICADO\?", "INDICACAO"),
        (r"^\bCOMPOSI[ÇC][ÃA]O\b", "COMPOSICAO"),
        (r"^(3|III)\.?\s*QUANDO N[ÃA]O DEVO USAR (ESTE )?MEDICAMENTO\?", "CONTRAINDICACAO"),
        (r"^(6|VI)\.?\s*COMO DEVO USAR (ESTE )?MEDICAMENTO\?", "POSOLOGIA"),
        (r"^\bPOSOLOGIA\b", "POSOLOGIA"),
        (r"^(8|VIII)\.?\s*QUAIS OS MALES QUE ESTE MEDICAMENTO PODE ME CAUSAR\?", "EFEITOS_ADVERSOS"),
        (r"^\bREA[ÇC][ÕO]ES? ADVERSAS\b", "EFEITOS_ADVERSOS"),
        
        # --- Nossas "Etiquetas de Parada" (voltam para OUTROS) ---
        (r"^(2|II)\.?\s*COMO ESTE MEDICAMENTO FUNCIONA\?", STOP_LABEL),
        (r"^(4|IV)\.?\s*O QUE DEVO SABER ANTES DE USAR (ESTE )?MEDICAMENTO\?", STOP_LABEL),
        (r"^(5|V)\.?\s*ONDE, COMO E POR QUANTO TEMPO POSSO GUARDAR (ESTE )?MEDICAMENTO\?", STOP_LABEL),
        (r"^(7|VII)\.?\s*O QUE DEVO FAZER QUANDO EU ME ESQUECER DE USAR (ESTE )?MEDICAMENTO\?", STOP_LABEL),
        (r"^(9|IX)\.?\s*O QUE FAZER SE ALGUÉM USAR UMA QUANTIDADE MAIOR DO QUE A INDICADA (DESTE )?MEDICAMENTO\?", STOP_LABEL),
        (r"^\bDIZERES LEGAIS\b", STOP_LABEL),
        (r"^\bAPRESENTA[ÇC][ÕO]ES\b", STOP_LABEL),
        (r"^\bINTERA[ÇC][ÕO]ES MEDICAMENTOSAS\b", STOP_LABEL),
    ]

    # Compila regex para eficiência
    regex_mapeada = [(re.compile(p, re.IGNORECASE), lbl) for p, lbl in padroes]

    dados = []
    buffer = []
    label_atual = "OUTROS" # Começa como OUTROS (para o cabeçalho/resumo inicial)

    def flush():
        """Salva o buffer de texto atual na lista de dados."""
        if not buffer:
            return
        bloco = "\n".join(buffer).strip()
        # Ignora blocos muito pequenos ou com pouco texto
        if len(bloco) >= 50: 
            dados.append((bloco, label_atual))
        buffer.clear() # Limpa o buffer

    # Varre linha a linha
    for linha in texto.splitlines():
        linha_limpa = linha.strip()
        
        # 1. Filtro de Lixo: Pula linhas vazias ou que são lixo óbvio
        if not linha_limpa or JUNK_REGEX.search(linha_limpa):
            continue

        encontrado = None
        for rx, lbl in regex_mapeada:
            # Verifica se a linha é um título
            if rx.search(linha_limpa):
                encontrado = lbl
                break
        
        if encontrado:
            # Novo cabeçalho encontrado: fecha o bloco anterior
            flush()
            # Se for um "Stop Label", volta para OUTROS. Senão, usa a nova etiqueta.
            label_atual = "OUTROS" if encontrado == STOP_LABEL else encontrado
            continue # Não adiciona o próprio título ao buffer

        buffer.append(linha) # Adiciona a linha de texto ao buffer atual

    # Salva o último bloco
    flush()

    # Estratégia adicional: quebrar blocos longos por parágrafos
    dados_expandidos = []
    for bloco, lbl in dados:
        # Quebra por 2+ quebras de linha (parágrafos)
        paragrafos = [p.strip() for p in re.split(r"(\n\s*){2,}", bloco) if p.strip()]
        
        for p in paragrafos:
            # Filtro final de limpeza
            if len(p) >= 50: # Garante que o parágrafo tenha conteúdo
                dados_expandidos.append((p, lbl))
    
    return dados_expandidos

# --- 3. EXECUÇÃO PRINCIPAL (ATUALIZADA) ---
def main():
    pasta_data = "data"
    pasta_dataset = "dataset"
    os.makedirs(pasta_dataset, exist_ok=True)
    
    # Pega TODOS os arquivos .pdf da pasta 'data'
    arquivos_pdf = [f for f in os.listdir(pasta_data) if f.endswith(".pdf")]
    
    if not arquivos_pdf:
        print(f"Nenhum arquivo .pdf encontrado na pasta '{pasta_data}'.")
        return

    print(f"--- Encontrados {len(arquivos_pdf)} PDFs. Iniciando etiquetagem automática... ---")
    
    todos_os_dados = []
    
    # Loop para processar cada PDF
    for nome_pdf in arquivos_pdf:
        caminho_pdf = os.path.join(pasta_data, nome_pdf)
        print(f"Processando: {nome_pdf}")
        
        texto = extrair_texto_pdf(caminho_pdf)
        if not texto:
            continue
            
        dados_etiquetados = segmentar_e_etiquetar(texto)
        print(f"  -> Gerou {len(dados_etiquetados)} exemplos.")
        todos_os_dados.extend(dados_etiquetados)

    print("\n--- Processamento concluído! ---")
    
    if not todos_os_dados:
        print("Nenhum dado foi gerado. Verifique seus PDFs.")
        return

    # Cria um único DataFrame com tudo
    df = pd.DataFrame(todos_os_dados, columns=["texto", "label"])

    # Salva o arquivo CSV final
    caminho_csv = os.path.join(pasta_dataset, "dataset_completo_automatico.csv")
    df.to_csv(caminho_csv, index=False)

    print(f"\nSucesso! {len(df)} exemplos no total salvos em: {caminho_csv}")
    print("\nAmostra do dataset completo:")
    print(df.sample(5)) # Mostra 5 exemplos aleatórios
    
    print("\nDistribuição das etiquetas (contagem de exemplos):")
    print(df['label'].value_counts()) # Mostra quantas de cada etiqueta


if __name__ == "__main__":
    main()