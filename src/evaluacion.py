import torch
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)

from src.configuracion import (NUM_CLASSES,IDX_TO_CLASS)


@torch.no_grad()
def evaluarM(model, dataloader, device):

    model.eval()

    y_true = []
    y_pred = []

    for pre_img, post_img, labels in dataloader:

        pre_img = pre_img.to(device,non_blocking=True)

        post_img = post_img.to(device,non_blocking=True)

        outputs = model(pre_img,post_img)

        preds = outputs.argmax(dim=1)

        y_true.extend(labels.numpy())

        y_pred.extend(preds.cpu().numpy())

    clases = list(
        range(NUM_CLASSES)
    )

    nombres = [
        IDX_TO_CLASS[i]
        for i in clases
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    precision_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    precision_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=clases
    )

    reporte = classification_report(
        y_true,
        y_pred,
        labels=clases,
        target_names=nombres,
        digits=4,
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision_weighted,
        "recall": recall_weighted,
        "f1": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "report": reporte
    }