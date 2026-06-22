import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src.configuracion import IMG_SIZE, CLASSES, PADDING
from src.lectura import leer_json, leer_recorte, obtener_archivos, obtener_edificios, obtener_coordenadas, obtener_bbox

#Estadisticas de normalizacion ImageNet
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

#Dataset que carga pares pre/post disaster con sus etiquetas de dano
class XView2(Dataset):

    #Inicializa el dataset cargando escenas y edificios
    def __init__(self, dataset_dir, sample_size=None, random_state=42):
        self.samples = []
        escenas = obtener_archivos(dataset_dir)
        print(f"Escenas encontradas: {len(escenas)}")
        for escena in escenas:
            json_data = leer_json(escena["post_json"])
            edificios = obtener_edificios(json_data)
            for edificio in edificios:
                etiqueta = edificio["label"]
                if etiqueta not in CLASSES:
                    continue
                coords = obtener_coordenadas(edificio["wkt"])
                bbox = obtener_bbox(coords)
                self.samples.append({
                    "scene_id": str(escena["pre_img"]),
                    "pre_img": escena["pre_img"],
                    "post_img": escena["post_img"],
                    "bbox": bbox,
                    "label": CLASSES[etiqueta]
                })
        print(f"Total edificios encontrados: {len(self.samples)}")
        # Muestreo estratificado opcional
        if sample_size is not None and sample_size < len(self.samples):
            indices = np.arange(len(self.samples))
            etiquetas = np.array([sample["label"] for sample in self.samples])
            seleccionados, _ = train_test_split(indices, train_size=sample_size, stratify=etiquetas, random_state=random_state)
            self.samples = [self.samples[i] for i in seleccionados]
        print(f"Total edificios utilizados: {len(self.samples)}")

    #Retorna la cantidad total de muestras
    def __len__(self):
        return len(self.samples)

    #Redimensiona, normaliza y convierte a tensor formato CHW
    def preparar_imagen(self, imagen):
        imagen = cv2.resize(imagen, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        imagen = np.clip(imagen, 0, 255)
        imagen = imagen.astype(np.float32) / 255.0
        imagen = np.ascontiguousarray(imagen)
        imagen = torch.from_numpy(imagen).permute(2, 0, 1)
        return (imagen - IMAGENET_MEAN) / IMAGENET_STD

    #Retorna (imagen_pre, imagen_post, etiqueta) para un edificio
    def __getitem__(self, idx):
        sample = self.samples[idx]
        pre_crop = leer_recorte(sample["pre_img"], sample["bbox"], PADDING)
        post_crop = leer_recorte(sample["post_img"], sample["bbox"], PADDING)
        pre_crop = self.preparar_imagen(pre_crop)
        post_crop = self.preparar_imagen(post_crop)
        label = torch.tensor(sample["label"], dtype=torch.long)
        return pre_crop, post_crop, label
