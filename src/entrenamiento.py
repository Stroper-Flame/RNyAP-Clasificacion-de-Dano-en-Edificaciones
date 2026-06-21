import copy
import numpy as np
import torch

from torch.utils.data import (
    DataLoader,
    Subset
)

from sklearn.model_selection import (
    GroupShuffleSplit
)

from src.configuracion import (
    BATCH_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    TEST_SIZE,
    RANDOM_STATE,
    NUM_WORKERS,
    NUM_CLASSES,
    DEVICE
)


def crear_loader(dataset, shuffle):

    parametros = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "pin_memory": DEVICE.type == "cuda"
    }

    if NUM_WORKERS > 0:

        parametros["persistent_workers"] = True
        parametros["prefetch_factor"] = 2

    return DataLoader(**parametros)


def crear_dataloaders(dataset):

    suma = (
        TRAIN_SIZE
        + VAL_SIZE
        + TEST_SIZE
    )

    if abs(suma - 1.0) > 0.000001:

        raise ValueError(
            "TRAIN_SIZE, VAL_SIZE y TEST_SIZE "
            "deben sumar 1.0"
        )

    etiquetas = np.array(
        [
            sample["label"]
            for sample in dataset.samples
        ]
    )

    escenas = np.array(
        [
            sample["scene_id"]
            for sample in dataset.samples
        ]
    )

    primer_split = GroupShuffleSplit(
        n_splits=1,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE
    )

    train_indices, temporal_indices = next(
        primer_split.split(
            np.zeros(
                (len(dataset), 1)
            ),
            etiquetas,
            escenas
        )
    )

    proporcion_val = (
        VAL_SIZE
        / (VAL_SIZE + TEST_SIZE)
    )

    segundo_split = GroupShuffleSplit(
        n_splits=1,
        train_size=proporcion_val,
        random_state=RANDOM_STATE
    )

    val_posiciones, test_posiciones = next(
        segundo_split.split(
            np.zeros(
                (len(temporal_indices), 1)
            ),
            etiquetas[temporal_indices],
            escenas[temporal_indices]
        )
    )

    val_indices = temporal_indices[
        val_posiciones
    ]

    test_indices = temporal_indices[
        test_posiciones
    ]

    train_dataset = Subset(
        dataset,
        train_indices.tolist()
    )

    val_dataset = Subset(
        dataset,
        val_indices.tolist()
    )

    test_dataset = Subset(
        dataset,
        test_indices.tolist()
    )

    train_loader = crear_loader(
        train_dataset,
        shuffle=True
    )

    val_loader = crear_loader(
        val_dataset,
        shuffle=False
    )

    test_loader = crear_loader(
        test_dataset,
        shuffle=False
    )

    conteos = np.bincount(
        etiquetas[train_indices],
        minlength=NUM_CLASSES
    ).astype(np.float32)

    if np.any(conteos == 0):

        raise ValueError(
            "Falta alguna clase en entrenamiento. "
            f"Conteos: {conteos.tolist()}"
        )

    pesos = (
        conteos.sum()
        / (NUM_CLASSES * conteos)
    )

    class_weights = torch.tensor(
        pesos,
        dtype=torch.float32
    )

    print(
        f"Train: {len(train_dataset)} muestras"
    )

    print(
        f"Val:   {len(val_dataset)} muestras"
    )

    print(
        f"Test:  {len(test_dataset)} muestras"
    )

    print(
        "Conteos de entrenamiento:",
        conteos.astype(int).tolist()
    )

    print(
        "Pesos calculados:",
        pesos.round(4).tolist()
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        class_weights
    )


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for pre_img, post_img, labels in dataloader:

        pre_img = pre_img.to(
            device,
            non_blocking=True
        )

        post_img = post_img.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            pre_img,
            post_img
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = outputs.argmax(
            dim=1
        )

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss / len(dataloader)
    )

    epoch_acc = correct / total

    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_epoch(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for pre_img, post_img, labels in dataloader:

        pre_img = pre_img.to(
            device,
            non_blocking=True
        )

        post_img = post_img.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(
            pre_img,
            post_img
        )

        loss = criterion(
            outputs,
            labels
        )

        running_loss += loss.item()

        preds = outputs.argmax(
            dim=1
        )

        correct += (
            preds == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = (
        running_loss / len(dataloader)
    )

    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    device,
    patience=10
):

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")

    best_state = copy.deepcopy(
        model.state_dict()
    )

    epochs_without_improvement = 0

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

        history["train_loss"].append(
            train_loss
        )

        history["train_acc"].append(
            train_acc
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_acc"].append(
            val_acc
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                "Early stopping en la época "
                f"{epoch + 1}"
            )

            break

    model.load_state_dict(
        best_state
    )

    print(
        "Mejor pérdida de validación: "
        f"{best_val_loss:.4f}"
    )

    return history