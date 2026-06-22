import sys
import glob as glob_mod
import matplotlib.pyplot as plt
import torch

from src.configuracion import IDX_TO_CLASS, DEVICE, MODELOS_DIR

#Estadisticas de normalizacion ImageNet (mismas que en dataset.py)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@torch.no_grad()
def mostrar_prediccion(model, dataset, idx, device):
    #Obtener la muestra del dataset
    pre_crop, post_crop, label = dataset[idx]

    #Denormalizar para visualizacion (CxHxC -> HxWxC)
    pre_img = pre_crop * IMAGENET_STD + IMAGENET_MEAN
    pre_img = pre_img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    post_img = post_crop * IMAGENET_STD + IMAGENET_MEAN
    post_img = post_img.permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    #Inferencia
    model.eval()
    pre_batch = pre_crop.unsqueeze(0).to(device, non_blocking=True)
    post_batch = post_crop.unsqueeze(0).to(device, non_blocking=True)
    output = model(pre_batch, post_batch)
    probs = torch.softmax(output, dim=1).squeeze(0).cpu()
    pred = probs.argmax().item()

    #Visualizacion con matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pre_img)
    axes[0].set_title("PRE - Desastre")
    axes[0].axis("off")
    axes[1].imshow(post_img)
    axes[1].set_title("POST - Desastre")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()

    #Imprimir resultados
    sample = dataset.samples[idx]
    real_label = IDX_TO_CLASS[label.item()]
    pred_label = IDX_TO_CLASS[pred]

    print("=" * 55)
    print(f"  Indice de muestra: {idx}")
    print(f"  Imagen PRE:        {sample['pre_img']}")
    print(f"  Imagen POST:       {sample['post_img']}")
    print(f"  Etiqueta real:     {real_label} ({label.item()})")
    print(f"  Prediccion:        {pred_label} ({pred})")
    print("-" * 55)
    print("  Probabilidades por clase:")
    for i, prob in enumerate(probs):
        print(f"    {IDX_TO_CLASS[i]:20s}: {prob:.4f}  ({prob*100:.2f}%)")
    print("=" * 55)

    return pred, probs

#Busca el archivo GPU_*.pth en MODELOS_DIR y retorna el modelo cargado
def cargar_modelo(device=None):
    if device is None:
        device = DEVICE

    modelos = sorted(glob_mod.glob(str(MODELOS_DIR / "GPU_*.pth")))
    if not modelos:
        raise FileNotFoundError(f"No se encontro ningun archivo GPU_*.pth en {MODELOS_DIR}")

    modelo_path = modelos[0]
    print(f"Cargando modelo: {modelo_path}")

    from src.modelos import SiameseResNet18
    checkpoint = torch.load(modelo_path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model = SiameseResNet18().to(device)
    model.load_state_dict(state_dict)
    print(f"Modelo cargado en {device}")
    return model


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Uso: python {sys.argv[0]} <indice>")
        sys.exit(1)

    idx = int(sys.argv[1])

    from src.dataset import XView2
    from src.configuracion import DATASET_DIR, SAMPLE_SIZE, RANDOM_STATE

    #Cargar dataset
    print("Cargando dataset...")
    dataset = XView2(DATASET_DIR, sample_size=SAMPLE_SIZE, random_state=RANDOM_STATE)

    if idx < 0 or idx >= len(dataset):
        print(f"Error: indice {idx} fuera de rango (0-{len(dataset)-1})")
        sys.exit(1)

    model = cargar_modelo()
    mostrar_prediccion(model, dataset, idx, DEVICE)