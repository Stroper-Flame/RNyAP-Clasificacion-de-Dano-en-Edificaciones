import matplotlib.pyplot as plt
import seaborn as sns

from src.configuracion import RESULTADOS_DIR

#Curva de perdida de entrenamiento y validacion
def graficar_loss(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss de Entrenamiento")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / "loss.png")
    plt.show()

#Curva de exactitud de entrenamiento y validacion
def graficar_accuracy(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy de Entrenamiento")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / "accuracy.png")
    plt.show()

#Heatmap de la matriz de confusion
def graficar_matriz_confusion(confusion_matrix, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    plt.title("Matriz de Confusion")
    plt.tight_layout()
    plt.savefig(RESULTADOS_DIR / "matriz_confusion.png")
    plt.show()

#Imprime en consola todas las metricas de evaluacion
def mostrar_metricas(resultados):
    for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1",
                "precision_macro", "recall_macro", "f1_macro"):
        print(f"{key:20s}: {resultados[key]:.4f}")
