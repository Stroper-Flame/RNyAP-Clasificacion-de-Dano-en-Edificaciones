import json
from pathlib import Path
import cv2
import numpy as np

def leer_json(json_path):
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        datos = json.load(f)
    return datos

def leer_imagen(img_path):
    img_path = str(img_path)
    imagen = cv2.imread(img_path)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo leer la imagen:\n{img_path}")
    imagen = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)
    return imagen

def obtener_archivos_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    muestras = []
    particiones = ["tier1","tier3","test","hold"]

    for particion in particiones:
        images_dir = dataset_dir / particion / "images"
        labels_dir = dataset_dir / particion / "labels"

        if not images_dir.exists():
            continue

        pre_imgs = sorted(images_dir.glob("*pre_disaster*.png"))

        if len(pre_imgs) == 0:
            pre_imgs = sorted(images_dir.glob("*pre_disaster*.tif"))

        for pre_img in pre_imgs:
            nombre = pre_img.name.replace("_pre_disaster","")
            post_img = images_dir / pre_img.name.replace("_pre_disaster","_post_disaster")
            pre_json = labels_dir / pre_img.name.replace(".png",".json").replace(".tif",".json")
            post_json = labels_dir / pre_json.name.replace("_pre_disaster","_post_disaster")

            if (
                post_img.exists()
                and pre_json.exists()
                and post_json.exists()
            ):

                muestras.append(
                    {
                        "id": nombre,
                        "pre_img": pre_img,
                        "post_img": post_img,
                        "pre_json": pre_json,
                        "post_json": post_json
                    }
                )

    return muestras


def mostrar_resumen_dataset(muestras):
    print(f"Total de muestras: {len(muestras):,}")

    if len(muestras) > 0:

        ejemplo = muestras[0]

        print("\nEjemplo:")
        print(f"ID        : {ejemplo['id']}")
        print(f"PRE IMG   : {ejemplo['pre_img']}")
        print(f"POST IMG  : {ejemplo['post_img']}")
        print(f"PRE JSON  : {ejemplo['pre_json']}")
        print(f"POST JSON : {ejemplo['post_json']}")