import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import optuna
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, TQDMProgressBar
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.classification import Accuracy, F1Score
from torchvision import models as tv_models

from train_improved import (
    ChestXRayDataModule
)


class OptunaChestXRayClassifier(pl.LightningModule):
    """Version du classifier pour Optuna avec hyperparamètres configurables"""

    def __init__(self, trial, num_classes=3, class_weights=None, pretrained_weights_path=None):
        super().__init__()

        # Hyperparamètres à optimiser
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        dropout1 = trial.suggest_float("dropout1", 0.2, 0.6)
        dropout2 = trial.suggest_float("dropout2", 0.1, 0.4)
        hidden_size = trial.suggest_int("hidden_size", 256, 512, step=128)
        unfreeze_epoch_layer4 = trial.suggest_int("unfreeze_epoch_layer4", 2, 8)
        unfreeze_epoch_layer3 = trial.suggest_int("unfreeze_epoch_layer3", 4, 12)
        lr_ratio = trial.suggest_float("lr_ratio", 0.01, 0.3)
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.15)
        scheduler_type = trial.suggest_categorical("scheduler", ["cosine", "onecycle"])

        self.save_hyperparameters()
        self.trial = trial
        self.unfreeze_epoch_layer4 = unfreeze_epoch_layer4
        self.unfreeze_epoch_layer3 = unfreeze_epoch_layer3
        self.lr_ratio = lr_ratio
        self.label_smoothing = label_smoothing
        self.scheduler_type = scheduler_type

        # Charger le modèle
        if pretrained_weights_path and os.path.exists(pretrained_weights_path):
            print(f"Trial {trial.number}: Chargement modèle pré-entraîné {pretrained_weights_path}...")
            model = tv_models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            # Charger les poids pré-entraînés (sans la tête qui sera remplacée)
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            if any(key.startswith("model.") for key in state_dict.keys()):
                state_dict = {key.replace("model.", "", 1): value for key, value in state_dict.items()}
            # Exclure la tête (fc) car on va la remplacer avec les nouveaux hyperparamètres
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc.')}
            model.load_state_dict(state_dict, strict=False)
            print(f"Trial {trial.number}: Backbone pré-entraîné chargé (tête sera remplacée)")
        else:
            print(f"Trial {trial.number}: Chargement ResNet18 ImageNet...")
            model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs = model.fc.in_features
            print(f"Trial {trial.number}: ResNet18 ImageNet chargé")

        # Tête avec hyperparamètres optimisés
        # ResNet18 a 512 features, ajuster hidden_size si nécessaire
        model.fc = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(num_ftrs, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden_size, num_classes)
        )

        # Geler le backbone initialement
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

        self.model = model

        # Loss avec class weights
        if class_weights:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            self.loss_fn = nn.CrossEntropyLoss(
                weight=class_weights_tensor,
                label_smoothing=label_smoothing
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average='macro')

        # Stocker les hyperparamètres pour l'optimiseur
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Dégeler progressivement
        if self.current_epoch == self.unfreeze_epoch_layer4:
            for param in self.model.layer4.parameters():
                param.requires_grad = True
        if self.current_epoch == self.unfreeze_epoch_layer3:
            for param in self.model.layer3.parameters():
                param.requires_grad = True

        self.log('train_loss', loss, on_step=False, on_epoch=True)
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
        }, on_step=False, on_epoch=True)

        return {'val_acc': acc, 'val_f1': f1}

    def configure_optimizers(self):
        backbone_params = [p for n, p in self.model.named_parameters() if 'fc' not in n]
        head_params = [p for n, p in self.model.named_parameters() if 'fc' in n]

        optimizer = optim.AdamW(
            [
                {'params': backbone_params, 'lr': self.learning_rate * self.lr_ratio},
                {'params': head_params, 'lr': self.learning_rate}
            ],
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        if self.scheduler_type == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[self.learning_rate * self.lr_ratio, self.learning_rate],
                total_steps=self.trainer.estimated_stepping_batches,
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

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval
            }
        }


class OptunaPruningCallback(pl.Callback):
    """Callback pour pruner les essais Optuna"""

    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        current_score = trainer.callback_metrics.get('val_acc')

        if current_score is not None:
            self.trial.report(current_score.item(), epoch)

            # Pruner si l'essai n'est pas prometteur
            if self.trial.should_prune():
                raise optuna.TrialPruned()


def objective(trial):
    """Fonction objectif pour Optuna"""

    # Fixer la seed pour reproductibilité
    pl.seed_everything(42)

    # Message pour le premier trial
    if trial.number == 0:
        print("Premier essai : chargement du modèle pré-entraîné...")

    try:
        # Hyperparamètres pour le DataModule
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        print(f"[Trial {trial.number}] batch_size={batch_size}")

        # Setup data
        print(f"[Trial {trial.number}] Setup des données...")
        data_dir = "./dataset/"
        dm = ChestXRayDataModule(data_dir, batch_size=batch_size)
        dm.setup()
        print(f"[Trial {trial.number}] Données chargées ({len(dm.train_dataset)} train, {len(dm.val_dataset)} val)")

        # Créer le modèle avec hyperparamètres du trial
        print(f"[Trial {trial.number}] Création du modèle...")
        pretrained_path = "./models/best_90_resnet18_chest_xray_classifier_weights.pth"
        model = OptunaChestXRayClassifier(
            trial=trial,
            num_classes=3,
            class_weights=dm.class_weights,
            pretrained_weights_path=pretrained_path
        )
        print(f"[Trial {trial.number}] Modèle créé")

        # Callbacks
        early_stop = EarlyStopping(
            monitor="val_acc",
            patience=4,
            mode="max",
            min_delta=0.001
        )

        optuna_callback = OptunaPruningCallback(trial)

        # Trainer avec moins d'époques pour l'optimisation
        if torch.backends.mps.is_available():
            precision = "32"
        else:
            precision = "16-mixed"

        print(f"[Trial {trial.number}] Configuration du trainer...")

        # Vérifier les dataloaders avant de créer le trainer
        print(f"[Trial {trial.number}] Vérification des dataloaders...")
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        print(f"[Trial {trial.number}] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # Tester un batch pour vérifier que ça fonctionne
        print(f"[Trial {trial.number}] Test d'un batch...")
        try:
            test_batch = next(iter(train_loader))
            print(f"[Trial {trial.number}] Batch test OK: {test_batch[0].shape}, {test_batch[1].shape}")
        except Exception as e:
            print(f"[Trial {trial.number}] ERREUR lors du test batch: {e}")
            raise

        progress_cb = TQDMProgressBar(refresh_rate=1)
        trainer = pl.Trainer(
            max_epochs=15,
            accelerator="auto",
            devices=1,
            precision=precision,
            callbacks=[early_stop, optuna_callback, progress_cb],
            logger=False,
            enable_progress_bar=True,
            enable_model_summary=False,
            enable_checkpointing=False,
            gradient_clip_val=1.0,
            accumulate_grad_batches=2,
            num_sanity_val_steps=0,
            limit_train_batches=0.3,
            limit_val_batches=0.3,
            log_every_n_steps=10
        )

        # Entraîner
        print(f"[Trial {trial.number}] Démarrage de l'entraînement...")
        trainer.fit(model, dm)
        print(f"[Trial {trial.number}] Entraînement terminé")

        # Retourner la meilleure validation accuracy
        best_score = trainer.callback_metrics.get('val_acc')
        if best_score is not None:
            score = best_score.item()
            print(f"[Trial {trial.number}] Score final: {score:.4f}")
            return score
        else:
            print(f"[Trial {trial.number}] Aucun score trouvé, retour 0.0")
            return 0.0
    except optuna.TrialPruned:
        # Trial arrêté par le pruner (normal, pas une erreur)
        print(f"[Trial {trial.number}] Trial pruné (arrêté car non prometteur)")
        raise  # Re-lancer pour qu'Optuna le gère correctement
    except Exception as e:
        print(f"[Trial {trial.number}] ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def run_optuna_optimization(n_trials=50, n_jobs=1):
    """Lance l'optimisation Optuna"""

    # Créer l'étude
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    )

    print(f"Démarrage de l'optimisation avec {n_trials} essais...")
    print("Note: Utilisation du modèle pré-entraîné best_90_resnet18_chest_xray_classifier_weights.pth")
    print()

    # Optimiser
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    # Afficher les résultats
    print("\n" + "="*50)
    print("OPTIMISATION TERMINÉE")
    print("="*50)

    best_trial = study.best_trial
    print(f"\nMeilleure précision: {best_trial.value:.4f}")
    print("\nMeilleurs hyperparamètres:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Sauvegarder les résultats
    import pandas as pd
    df = study.trials_dataframe()
    df.to_csv("./optuna_results.csv", index=False)
    print(f"\nRésultats sauvegardés dans optuna_results.csv")

    return study, best_trial


if __name__ == "__main__":
    # Lancer l'optimisation
    study, best_trial = run_optuna_optimization(n_trials=20, n_jobs=1)

    print("\n" + "="*50)
    print("Pour utiliser les meilleurs hyperparamètres:")
    print("="*50)
    print("\nCopiez ces valeurs dans train_improved.py:")
    for key, value in best_trial.params.items():
        print(f"  {key} = {value}")
