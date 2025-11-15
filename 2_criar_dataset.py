import os
import re
import pandas as pd
import fitz  # PyMuPDF


def extrair_texto_pdf(caminho_pdf: str) -> str:
    texto = ""
    with fitz.open(caminho_pdf) as doc:
        for pagina in doc:
            texto += pagina.get_text()
    return texto


def segmentar_e_etiquetar(texto: str):
    # Normaliza para facilitar a detecção de seções
    txt = texto.upper()

    # Padrões de cabeçalhos típicos de bula ANVISA mapeados para nossos rótulos
    padroes = [
        (r"\bCOMPOSI[ÇC][ÃA]O\b", "COMPOSICAO"),
        (r"\bINDICA[ÇC][ÃA]O(ES)?\b|\bPARA QUE ESTE MEDICAMENTO( É| E)? INDICADO\b", "INDICACAO"),
        (r"\bCONTRAINDICA[ÇC][ÃA]O(ES)?\b|\bQUANDO N[ÃA]O DEVO USAR\b", "CONTRAINDICACAO"),
        (r"\bPOSOLOGIA\b|\bCOMO DEVO USAR\b|\bMODO DE USAR\b|\bCOMO USAR\b|\bPOSOLOGIA E MODO DE USAR\b", "POSOLOGIA"),
        (r"\bREA[ÇC][ÕO]ES? ADVERSAS\b|\bEFEITOS? ADVERSOS\b|\bREA[ÇC][ÕO]ES INDESEJADAS\b|\bQUAIS OS MALES\b", "EFEITOS_ADVERSOS"),
    ]

    # Compila regex para eficiência
    regex_mapeada = [(re.compile(p, re.IGNORECASE), lbl) for p, lbl in padroes]

    dados = []
    buffer = []
    label_atual = "OUTROS"

    def flush():
        if not buffer:
            return
        bloco = "\n".join(buffer).strip()
        # Ignora blocos muito pequenos
        if len(bloco) >= 50:
            dados.append((bloco, label_atual))

    # Varre linha a linha, trocando o rótulo quando detectar cabeçalho
    for linha in txt.splitlines():
        linha_limpa = linha.strip()
        if not linha_limpa:
            buffer.append("")
            continue

        encontrado = None
        for rx, lbl in regex_mapeada:
            if rx.search(linha_limpa):
                encontrado = lbl
                break

        if encontrado:
            # novo cabeçalho encontrado: fecha o bloco anterior
            flush()
            buffer = []
            label_atual = encontrado
            continue

        buffer.append(linha)

    # Último bloco
    flush()

    # Estratégia adicional: quebrar blocos longos por parágrafos para gerar mais exemplos
    dados_expandidos = []
    for bloco, lbl in dados:
        paragrafos = [p.strip() for p in re.split(r"\n\s*\n+", bloco) if p.strip()]
        # Se houver parágrafos, adiciona cada um como exemplo; caso contrário, adiciona o bloco inteiro
        if len(paragrafos) > 1:
            for p in paragrafos:
                if len(p) >= 50:
                    dados_expandidos.append((p, lbl))
        else:
            dados_expandidos.append((bloco, lbl))

    # Garante pelo menos algum conteúdo; caso nada tenha sido categorizado, coloca tudo como OUTROS
    if not dados_expandidos and texto.strip():
        dados_expandidos.append((texto.strip(), "OUTROS"))

    return dados_expandidos


def main():
    nome_pdf = "bula_losartana.pdf"
    caminho_pdf = os.path.join("data", nome_pdf)
    print(f"--- Lendo PDF: {caminho_pdf} ---")

    if not os.path.exists(caminho_pdf):
        print("Arquivo não encontrado. Coloque o PDF em 'data/'.")
        return

    try:
        texto = extrair_texto_pdf(caminho_pdf)
    except Exception as e:
        print("Falha ao abrir/ler o PDF:", e)
        return

    print("Texto extraído. Iniciando segmentação e etiquetagem...")
    dados_etiquetados = segmentar_e_etiquetar(texto)

    print(f"Total de exemplos gerados: {len(dados_etiquetados)}")

    df = pd.DataFrame(dados_etiquetados, columns=["texto", "label"])

    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)
    caminho_csv = os.path.join(output_dir, "bula_losartana_etiquetada.csv")
    df.to_csv(caminho_csv, index=False)

    print(f"Sucesso! Dataset salvo em: {caminho_csv}")
    print("Amostra:")
    print(df.head())


if __name__ == "__main__":
    main()
