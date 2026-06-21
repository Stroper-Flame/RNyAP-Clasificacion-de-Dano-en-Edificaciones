import torch
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)

@torch.no_grad()
def evaluarM(model,dataloader,device):
    model.eval()
    y_true = []
    y_pred = []
    for pre_img, post_img, labels in dataloader:
        pre_img = pre_img.to(device)
        post_img = post_img.to(device)
        outputs = model(pre_img,post_img)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred,average="weighted",zero_division=0)

    recall = recall_score(y_true,y_pred,average="weighted",zero_division=0)
    f1 = f1_score(y_true,y_pred,average="weighted",zero_division=0)
    cm = confusion_matrix(y_true,y_pred)
    reporte = classification_report(y_true,y_pred,zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "report": reporte
    }