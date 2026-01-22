import warnings
from collections import Counter
warnings.filterwarnings("ignore", category=UserWarning)

import os

import lightning.pytorch as pl
from lightning.pytorch.callbacks import TQDMProgressBar
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, F1Score
from torchvision import datasets, transforms
from torchvision import models as tv_models

import utils.helper_utils as helper_utils

torch.set_float32_matmul_precision('medium')

data_dir = "./dataset/"

# Data augmentation plus contrôlée pour radiographies
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def create_datasets(train_path, val_path, train_transform, val_transform):
    train_dataset = datasets.ImageFolder(train_path, train_transform)
    val_dataset = datasets.ImageFolder(val_path, val_transform)
    return train_dataset, val_dataset


def get_class_weights(dataset):
    """Calcule les poids de classe pour gérer le déséquilibre"""
    class_counts = Counter([label for _, label in dataset.samples])
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = {i: total / (num_classes * count) for i, count in enumerate(class_counts.values())}
    return [weights[i] for i in range(num_classes)]


def load_dataloader(dataset, batch_size, is_train_loader, class_weights=None):
    if is_train_loader and class_weights:
        weights = [class_weights[label] for _, label in dataset.samples]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,  # 0 pour éviter problèmes multiprocessing sur macOS
            pin_memory=False
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train_loader,
            num_workers=0,  # 0 pour éviter problèmes multiprocessing sur macOS
            pin_memory=False
        )
    return loader


class ChestXRayDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transform = TRAIN_TRANSFORM
        self.val_transform = VAL_TRANSFORM
        self.train_dataset = None
        self.val_dataset = None
        self.class_weights = None

    def setup(self, stage=None):
        train_path = os.path.join(self.data_dir, "train")
        val_path = os.path.join(self.data_dir, "val")

        self.train_dataset, self.val_dataset = create_datasets(
            train_path, val_path, self.train_transform, self.val_transform
        )

        # Calculer les poids de classe
        self.class_weights = get_class_weights(self.train_dataset)
        print(f"Class weights: {self.class_weights}")

    def train_dataloader(self):
        return load_dataloader(
            self.train_dataset, self.batch_size, True, self.class_weights
        )

    def val_dataloader(self):
        return load_dataloader(
            self.val_dataset, self.batch_size, False
        )


def load_resnet18(num_classes, weights_path=None, dropout1=0.3, dropout2=0.2, hidden_size=512):
    """Charge ResNet18 avec tête améliorée"""
    # Charger le modèle pré-entraîné si fourni, sinon ImageNet
    if weights_path and os.path.exists(weights_path):
        model = tv_models.resnet18(weights=None)
        num_ftrs = model.fc.in_features

        # Charger les poids pré-entraînés (.pth = state_dict direct)
        state_dict = torch.load(weights_path, map_location='cpu')

        # Remplacer la tête
        mid_size = max(128, hidden_size // 2)
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden_size, mid_size),
            nn.BatchNorm1d(mid_size),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(mid_size, num_classes)
        )

        # Exclure la tête (fc) car elle a été remplacée
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
        model.load_state_dict(state_dict, strict=False)
    else:
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = model.fc.in_features
        mid_size = max(128, hidden_size // 2)
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden_size, mid_size),
            nn.BatchNorm1d(mid_size),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(mid_size, num_classes)
        )

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def define_optimizer_and_scheduler(model, learning_rate, weight_decay, num_epochs, total_steps, lr_ratio=0.1, scheduler_type="onecycle"):
    """Optimiseur avec scheduler configurable"""
    backbone_params = [p for n, p in model.named_parameters() if 'fc' not in n]
    head_params = [p for n, p in model.named_parameters() if 'fc' in n]

    optimizer = optim.AdamW(
        [
            {'params': backbone_params, 'lr': learning_rate * lr_ratio},
            {'params': head_params, 'lr': learning_rate}
        ],
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    if scheduler_type == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[learning_rate * lr_ratio, learning_rate],
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
        interval = "step"
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        interval = "epoch"

    return optimizer, scheduler, interval


class ChestXRayClassifier(pl.LightningModule):
    def __init__(self, model_weights_path=None, num_classes=3, learning_rate=1e-3,
                 weight_decay=1e-4, class_weights=None, dropout1=0.3, dropout2=0.2,
                 hidden_size=512, lr_ratio=0.1, label_smoothing=0.1,
                 scheduler_type="onecycle"):
        super().__init__()
        self.save_hyperparameters()

        self.model = load_resnet18(
            self.hparams.num_classes,
            self.hparams.model_weights_path,
            dropout1=self.hparams.dropout1,
            dropout2=self.hparams.dropout2,
            hidden_size=self.hparams.hidden_size
        )

        # Loss avec class weights
        if class_weights:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=self.hparams.num_classes, average='macro')
        self.lr_ratio = lr_ratio
        self.scheduler_type = scheduler_type

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        f1 = self.f1(logits, y)

        self.log_dict({
            'val_loss': loss,
            'val_acc': acc,
            'val_f1': f1
        }, prog_bar=True)

        return {'val_loss': loss, 'val_acc': acc}

    def configure_optimizers(self):
        optimizer, scheduler, interval = define_optimizer_and_scheduler(
            self.model,
            self.hparams.learning_rate,
            self.hparams.weight_decay,
            self.trainer.max_epochs,
            total_steps=self.trainer.estimated_stepping_batches,
            lr_ratio=self.lr_ratio,
            scheduler_type=self.scheduler_type
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval
            }
        }


def early_stopping(num_epochs, stop_threshold):
    stop = EarlyStopping(
        monitor="val_acc",
        stopping_threshold=stop_threshold,
        patience=10,  # Plus de patience
        mode="max",
        min_delta=0.001
    )
    return stop


class SavePthCheckpoint(Callback):
    """Callback pour sauvegarder les checkpoints au format .pth"""

    def __init__(self, dirpath="./checkpoints", monitor="val_acc", mode="max", save_top_k=3):
        super().__init__()
        self.dirpath = dirpath
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        os.makedirs(dirpath, exist_ok=True)
        self.best_scores = []  # Liste de (score, epoch, path)

    def on_validation_epoch_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return

        os.makedirs(self.dirpath, exist_ok=True)

        current_score = current_score.item()
        epoch = trainer.current_epoch
        train_loss = trainer.callback_metrics.get("train_loss")
        train_loss_val = train_loss.item() if train_loss is not None else None

        # Sauvegarder le meilleur modèle
        self.best_scores.append((current_score, epoch, None))
        self.best_scores.sort(reverse=(self.mode == "max"))

        # Garder seulement les top_k
        if len(self.best_scores) > self.save_top_k:
            self.best_scores = self.best_scores[:self.save_top_k]

        # Sauvegarder les top_k modèles
        for idx, (score, ep, _) in enumerate(self.best_scores):
            loss_part = f"-loss{train_loss_val:.4f}" if train_loss_val is not None else ""
            path = os.path.join(self.dirpath, f"best-{ep}-acc{score:.4f}{loss_part}.pth")
            if ep == epoch:  # Sauvegarder seulement le modèle de l'epoch actuelle
                torch.save(pl_module.model.state_dict(), path)
                self.best_scores[idx] = (score, ep, path)

        # Sauvegarder le dernier modèle
        last_path = os.path.join(self.dirpath, "last_model.pth")
        torch.save(pl_module.model.state_dict(), last_path)


def run_training(model, data_module, num_epochs, callback, progress_bar=True):
    if torch.backends.mps.is_available():
        precision = "32"
    else:
        precision = "16-mixed"

    pth_cb = SavePthCheckpoint(
        dirpath="./checkpoints",
        monitor="val_acc",
        mode="max",
        save_top_k=3
    )

    progress_cb = TQDMProgressBar(refresh_rate=1)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        devices=1,
        precision=precision,
        callbacks=[callback, pth_cb, progress_cb],
        logger=False,
        enable_progress_bar=progress_bar,
        enable_model_summary=False,
        enable_checkpointing=True,
        gradient_clip_val=1.0,  # Gradient clipping
        accumulate_grad_batches=2  # Gradient accumulation pour batch effectif plus grand
    )

    trainer.fit(model, data_module)
    return trainer, model


if __name__ == "__main__":
    # Configuration améliorée avec hyperparamètres optimisés par Optuna
    pl.seed_everything(42)

    # Hyperparamètres Optuna - Trial 2
    batch_size = 64
    learning_rate = 2.1930485556643678e-05
    weight_decay = 1.3492834268013232e-05
    dropout1 = 0.5795542149013333
    dropout2 = 0.38968960992236784
    hidden_size = 512
    lr_ratio = 0.2084275776885255
    label_smoothing = 0.06602287406094019
    scheduler_type = "onecycle"

    dm = ChestXRayDataModule(data_dir, batch_size=batch_size)
    dm.setup()

    model = ChestXRayClassifier(
        model_weights_path="../models/best_90_resnet18_chest_xray_classifier_weights.pth",
        num_classes=3,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        class_weights=dm.class_weights,
        dropout1=dropout1,
        dropout2=dropout2,
        hidden_size=hidden_size,
        lr_ratio=lr_ratio,
        label_smoothing=label_smoothing,
        scheduler_type=scheduler_type
    )

    early_stopping_callback = early_stopping(30, 0.99)

    trained_trainer, trained_model = run_training(
        model, dm, 30, early_stopping_callback
    )

    # Sauvegarder le meilleur modèle (déjà en .pth grâce au callback)
    # Trouver le meilleur checkpoint .pth
    import glob
    pth_files = glob.glob("./checkpoints/best-*.pth")
    if pth_files:
        # Trier par nom (qui contient l'accuracy) pour trouver le meilleur
        best_pth = max(pth_files, key=lambda x: float(x.split('-')[2].replace('.pth', '')))
        print(f"Meilleur modèle sauvegardé: {best_pth}")
        # Copier vers models/ si besoin
        import shutil
        shutil.copy(best_pth, "./models/resnet18_chest_xray_best.pth")
        print("Modèle copié vers ./models/resnet18_chest_xray_best.pth")

    # Évaluation finale
    helper_utils.per_class_acc_and_conf_matrix(trained_model, dm)
