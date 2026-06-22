from pathlib import Path
import torch

#Rutas de directorios del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = ROOT_DIR / "modelos"
RESULTADOS_DIR = ROOT_DIR / "resultados"
MODELOS_DIR.mkdir(parents=True, exist_ok=True)
RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)

#Ruta del dataset xView2
DATASET_DIR = Path("/home/markusancestro/Documentos/UPIIT-IPN/xView2_Dataset/geotiffs")

#None usa todas las muestras disponibles
SAMPLE_SIZE = 100000

#Mapeo clase a indice
CLASSES = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}
#Mapeo inverso indice a clase
IDX_TO_CLASS = {v: k for k, v in CLASSES.items()}
NUM_CLASSES = len(CLASSES)

#Preprocesamiento de imagenes
IMG_SIZE = 224
PADDING = 40

#Proporciones train/val/test
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

#DataLoader
BATCH_SIZE = 64
NUM_WORKERS = 4

#Backbone ResNet
BACKBONE = "resnet18"
PRETRAINED = True
FEATURE_DIM = 512

#Hiperparametros de entrenamiento
EPOCHS = 100
PATIENCE = 10
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

#Auto-detecta CUDA o CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
