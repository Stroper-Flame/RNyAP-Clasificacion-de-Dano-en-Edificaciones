import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit
from src.configuracion import BATCH_SIZE, TRAIN_SIZE, VAL_SIZE, TEST_SIZE, RANDOM_STATE, NUM_WORKERS, NUM_CLASSES, DEVICE

#Crea un DataLoader con la configuracion del proyecto
def crear_loader(dataset, shuffle):
    params = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda"
    }
    if NUM_WORKERS > 0:
        params["persistent_workers"] = True
        params["prefetch_factor"] = 2
    return DataLoader(**params)

#Divide el dataset en train/val/test usando GroupShuffleSplit por escena
def crear_dataloaders(dataset):
    assert abs(TRAIN_SIZE + VAL_SIZE + TEST_SIZE - 1.0) < 1e-6, "Las proporciones deben sumar 1.0"

    etiquetas = np.array([s["label"] for s in dataset.samples])
    escenas = np.array([s["scene_id"] for s in dataset.samples])

    # 1er split: train vs (val+test)
    gss1 = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=RANDOM_STATE)
    train_idx, temp_idx = next(gss1.split(np.zeros((len(dataset), 1)), etiquetas, escenas))

    # 2do split: val vs test dentro del temporal
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_ratio, random_state=RANDOM_STATE)
    val_idx, test_idx = next(gss2.split(np.zeros((len(temp_idx), 1)), etiquetas[temp_idx], escenas[temp_idx]))
    val_idx, test_idx = temp_idx[val_idx], temp_idx[test_idx]

    train_loader = crear_loader(Subset(dataset, train_idx.tolist()), shuffle=True)
    val_loader = crear_loader(Subset(dataset, val_idx.tolist()), shuffle=False)
    test_loader = crear_loader(Subset(dataset, test_idx.tolist()), shuffle=False)

    # Pesos para balanceo de clases
    conteos = np.bincount(etiquetas[train_idx], minlength=NUM_CLASSES).astype(np.float32)
    if np.any(conteos == 0):
        raise ValueError(f"Falta alguna clase en entrenamiento. Conteos: {conteos.tolist()}")
    pesos = conteos.sum() / (NUM_CLASSES * conteos)
    class_weights = torch.tensor(pesos, dtype=torch.float32)

    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print("Conteos:", conteos.astype(int).tolist(), "| Pesos:", pesos.round(4).tolist())
    return train_loader, val_loader, test_loader, class_weights

#Una epoca de entrenamiento. Retorna (loss, accuracy)
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = correct = total = 0.0

    for pre_img, post_img, labels in dataloader:
        pre_img = pre_img.to(device, non_blocking=True)
        post_img = post_img.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(pre_img, post_img)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), correct / total

#Una epoca de validacion sin gradients. Retorna (loss, accuracy)
@torch.no_grad()
def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = correct = total = 0.0

    for pre_img, post_img, labels in dataloader:
        pre_img = pre_img.to(device, non_blocking=True)
        post_img = post_img.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(pre_img, post_img)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), correct / total

#Bucle principal de entrenamiento con early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, patience=10):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stall = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
            stall = 0
        else:
            stall += 1

        if stall >= patience:
            print(f"Early stopping en epoca {epoch+1}")
            break

    model.load_state_dict(best_state)
    print(f"Mejor val_loss: {best_val_loss:.4f}")
    return history
