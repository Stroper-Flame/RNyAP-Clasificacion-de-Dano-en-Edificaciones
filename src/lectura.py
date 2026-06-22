import json
from rasterio.windows import Window
from pathlib import Path
import rasterio
import numpy as np
from shapely import wkt

#Cargar el archivo JSON con los datos
def leer_json(json_path):
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

#Leer una imagen .tiff y la retorna en formato HWC.
def leer_imagen(img_path):
    with rasterio.open(img_path) as src:
        imagen = src.read()
    return np.transpose(imagen, (1, 2, 0))

#Recortar la región de interés en la imagen alrededor del bounding box.
def leer_recorte(img_path, bbox, padding=0):
    xmin, ymin, xmax, ymax = bbox
    with rasterio.open(img_path) as src:
        x0 = min(max(0, int(xmin) - padding), src.width - 1)
        y0 = min(max(0, int(ymin) - padding), src.height - 1)
        x1 = min(max(x0 + 1, int(xmax) + padding), src.width)
        y1 = min(max(y0 + 1, int(ymax) + padding), src.height)
        ventana = Window(col_off=x0, row_off=y0, width=x1 - x0, height=y1 - y0)
        imagen = src.read(window=ventana)
    imagen = np.transpose(imagen, (1, 2, 0))
    # Asegurar 3 canales RGB
    if imagen.shape[2] > 3:
        imagen = imagen[:, :, :3]
    elif imagen.shape[2] == 1:
        imagen = np.repeat(imagen, 3, axis=2)
    return imagen

#Seleccionar los pares pre/post disaster y sus JSONs en todas las particiones.
def obtener_archivos(dataset_dir):
    dataset_dir = Path(dataset_dir)
    muestras = []
    particiones = ["tier1", "tier3", "test", "hold"]

    for particion in particiones:
        images_dir = dataset_dir / particion / "images"
        labels_dir = dataset_dir / particion / "labels"
        if not images_dir.exists():
            continue
        pre_imgs = sorted(images_dir.glob("*pre_disaster*.png"))
        if len(pre_imgs) == 0:
            pre_imgs = sorted(images_dir.glob("*pre_disaster*.tif"))
        for pre_img in pre_imgs:
            nombre = pre_img.name.replace("_pre_disaster", "")
            post_img = images_dir / pre_img.name.replace("_pre_disaster", "_post_disaster")
            pre_json = labels_dir / pre_img.name.replace(".png", ".json").replace(".tif", ".json")
            post_json = labels_dir / pre_json.name.replace("_pre_disaster", "_post_disaster")
            if post_img.exists() and pre_json.exists() and post_json.exists():
                muestras.append({"id": nombre, "pre_img": pre_img, "post_img": post_img, "pre_json": pre_json, "post_json": post_json})
    return muestras

#Imprime un resumen de las muestras encontradas
def resumen(muestras):
    print(f"Total de muestras: {len(muestras):,}")
    if len(muestras) > 0:
        ejemplo = muestras[0]
        print("\nEjemplo:")
        for key in ("id", "pre_img", "post_img", "pre_json", "post_json"):
            print(f"{key.upper():10s}: {ejemplo[key]}")

#Extrae lista de edificios
def obtener_edificios(json_data):
    edificios = []
    for feature in json_data["features"]["xy"]:
        props = feature["properties"]
        edificios.append({"uid": props["uid"], "label": props["subtype"], "wkt": feature["wkt"]})
    return edificios

#Convierte un polígono WKT a un arreglo de coordenadas
def obtener_coordenadas(wkt_polygon):
    poligono = wkt.loads(wkt_polygon)
    return np.array(poligono.exterior.coords)

#Calcula el bounding box a partir de las coordenadas
def obtener_bbox(coords):
    xmin = int(np.min(coords[:, 0]))
    xmax = int(np.max(coords[:, 0]))
    ymin = int(np.min(coords[:, 1]))
    ymax = int(np.max(coords[:, 1]))
    return xmin, ymin, xmax, ymax

#Recortar la imagen usando la bounding box
def recortar_imagen(imagen, bbox):
    xmin, ymin, xmax, ymax = bbox
    return imagen[ymin:ymax, xmin:xmax]