import matplotlib.pyplot as plt
import seaborn as sns

from src.configuracion import RESULTADOS_DIR


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


def graficar_matriz_confusion(confusion_matrix, class_names):

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")

    plt.tight_layout()

    plt.savefig(RESULTADOS_DIR / "matriz_confusion.png")
    plt.show()


def mostrar_metricas(resultados):

    print(
        f"Accuracy          : "
        f"{resultados['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy : "
        f"{resultados['balanced_accuracy']:.4f}"
    )

    print(
        f"Precision weighted: "
        f"{resultados['precision']:.4f}"
    )

    print(
        f"Recall weighted   : "
        f"{resultados['recall']:.4f}"
    )

    print(
        f"F1 weighted       : "
        f"{resultados['f1']:.4f}"
    )

    print(
        f"Precision macro   : "
        f"{resultados['precision_macro']:.4f}"
    )

    print(
        f"Recall macro      : "
        f"{resultados['recall_macro']:.4f}"
    )

    print(
        f"F1 macro          : "
        f"{resultados['f1_macro']:.4f}"
    )