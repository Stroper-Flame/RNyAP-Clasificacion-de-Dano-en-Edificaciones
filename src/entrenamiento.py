import torch
import torch.nn as nn
from torch.utils.data import (DataLoader,random_split)
from src.configuracion import (BATCH_SIZE,TRAIN_SIZE,VAL_SIZE,TEST_SIZE,RANDOM_STATE,NUM_WORKERS,DEVICE)

def crear_dataloaders(dataset):

    total = len(dataset)
    train_size = int(TRAIN_SIZE * total)
    val_size = int(VAL_SIZE * total)
    test_size = (total- train_size- val_size)
    generator = torch.Generator().manual_seed(RANDOM_STATE)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    return (train_loader,val_loader,test_loader)

def train_epoch(model,dataloader,criterion,optimizer,device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for pre_img, post_img, labels in dataloader:
        pre_img = pre_img.to(device)
        post_img = post_img.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(pre_img,post_img)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = (running_loss/ len(dataloader))
    epoch_acc = (correct/ total)
    return (epoch_loss,epoch_acc)

@torch.no_grad()
def validate_epoch(model,dataloader,criterion,device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    for pre_img, post_img, labels in dataloader:
        pre_img = pre_img.to(device)
        post_img = post_img.to(device)
        labels = labels.to(device)
        outputs = model(pre_img,post_img)
        loss = criterion(outputs,labels)
        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = (running_loss/ len(dataloader))
    epoch_acc = (correct/ total)
    return (epoch_loss,epoch_acc)

def train_model(model,train_loader,val_loader,criterion,optimizer,epochs,device):

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(epochs):

        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return history