from pathlib import Path

#Rutas de archivos a usar
ROOT_DIR = Path(__file__).resolve().parent.parent
CROPS_DIR = ROOT_DIR / "crops"
MODELOS_DIR = ROOT_DIR / "modelos"
RESULTADOS_DIR = ROOT_DIR / "resultados"

#Ruta para el Dataset
DATASET_DIR = Path("/home/markusancestro/Documentos/UPIIT-IPN/xView2_Dataset/geotiffs/")

#Clases en los JSON
CLASSES = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3
}
IDX_TO_CLASS = {
    0: "no-damage",
    1: "minor-damage",
    2: "major-damage",
    3: "destroyed"
}
NUM_CLASSES = len(CLASSES)

##Reducir las imagenes
IMG_SIZE = 224
PADDING = 40

#División del dataset
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

#Dataloader
BATCH_SIZE = 32
NUM_WORKERS = 4

#MOdedlo a usar
BACKBONE = "resnet18"
PRETRAINED = True
FEATURE_DIM = 512

#Parametros de Entrenamiento
EPOCHS = 1
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001

#Hadware a usar 
DEVICE = "cpu"