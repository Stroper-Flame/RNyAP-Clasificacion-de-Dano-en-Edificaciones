
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from src.configuracion import (IMG_SIZE,CLASSES,PADDING)
from src.lectura import (
    leer_json,
    leer_imagen,
    obtener_archivos,
    obtener_edificios,
    obtener_coordenadas,
    obtener_bbox
)

class XView2(Dataset):

    def __init__(self, dataset_dir):

        self.samples = []
        escenas = obtener_archivos(dataset_dir)
        print(f"Escenas encontradas: {len(escenas)}")
        for escena in escenas:
            json_data = leer_json(escena["post_json"])
            edificios = obtener_edificios(json_data)

            for edificio in edificios:
                if edificio["label"] == "un-classified":
                    continue
                coords = obtener_coordenadas(edificio["wkt"])
                bbox = obtener_bbox(coords)
                self.samples.append(
                    {
                        "pre_img": escena["pre_img"],
                        "post_img": escena["post_img"],
                        "bbox": bbox,
                        "label": CLASSES[edificio["label"]]
                    }
                )

        print(f"Total edificios: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pre_img = leer_imagen(sample["pre_img"])
        post_img = leer_imagen(sample["post_img"])
        xmin, ymin, xmax, ymax = sample["bbox"]

        h, w, _ = pre_img.shape

        xmin = max(0, xmin - PADDING)
        ymin = max(0, ymin - PADDING)

        xmax = min(w, xmax + PADDING)
        ymax = min(h, ymax + PADDING)

        pre_crop = pre_img[ ymin:ymax,xmin:xmax]
        post_crop = post_img[ymin:ymax,xmin:xmax]
        pre_crop = cv2.resize(pre_crop,(IMG_SIZE, IMG_SIZE))
        post_crop = cv2.resize(post_crop,(IMG_SIZE, IMG_SIZE))
        pre_crop = (pre_crop.astype(np.float32) / 255.0)
        post_crop = (post_crop.astype(np.float32) / 255.0)
        pre_crop = torch.tensor(pre_crop).permute(2, 0, 1)
        post_crop = torch.tensor(post_crop).permute(2, 0, 1)
        label = torch.tensor(sample["label"],dtype=torch.long)

        return (pre_crop,post_crop,label)