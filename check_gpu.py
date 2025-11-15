import torch

if torch.cuda.is_available():
    print("\n--- SUCESSO! ---")
    print(f"Placa de Vídeo detectada: {torch.cuda.get_device_name(0)}")
    print("Seu PyTorch está rodando na GPU (CUDA)!\n")
else:
    print("\n--- FALHA ---")
    print("O PyTorch NÃO conseguiu detectar sua GPU.")
    print("Tente reinstalar os drivers da NVIDIA ou o PyTorch.\n")