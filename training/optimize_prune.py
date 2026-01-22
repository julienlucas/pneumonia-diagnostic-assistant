import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models as tv_models
from train import ChestXRayDataModule, data_dir
import utils.helper_utils as helper_utils


def _iter_prunable_modules(model):
    """
    Itère sur les modules éligibles pour l'élagage.

    Yields
    ------
    Tuple[str, nn.Module]
        Paires de (nom de module pleinement qualifié, module) pour les couches
        élagables dans cet exercice : `nn.Conv2d` et `nn.Linear`.

    Notes
    -----
    - Le nom qualifié provient de `model.named_modules()` et reflète le
      chemin dans la hiérarchie du module (ex. "block.0", "classifier.fc").
    - Utilisez ce générateur pour appliquer systématiquement l'élagage sur le modèle.
    """
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            yield name, m


def finalize_pruning(model):
    """
    Rend l'élagage permanent en supprimant les wrappers de réparamétrage.

    Cela convertit tout paramètre élagué de la réparamétrisation (`weight_orig`, `weight_mask`)
    en un `weight` `nn.Parameter` régulier où les
    zéros sont **matérialisés** dans le tenseur stocké.
    """
    for _, module in _iter_prunable_modules(model):
        # Ne supprimer que si le paramètre a été élagué
        if hasattr(module, "weight_orig") and hasattr(module, "weight_mask"):
            prune.remove(module, "weight")
    return model


def sparsity_report(model):
    """
    Calcule la sparsité par couche et globale (fraction de zéros) sur les poids Conv2d/Linear.
    Returns
    -------
    dict
        {
          "layers": { "<name>.weight": 0.52, ... },
          "global_sparsity": 0.47
        }
    """
    total = 0
    zeros = 0
    for _, module in _iter_prunable_modules(model):
        if not hasattr(module, "weight"):
            continue
        weight = module.weight.detach()
        total += weight.numel()
        zeros += (weight == 0).sum().item()
    global_sparsity = (zeros / total) if total else 0.0
    return {"global_sparsity": global_sparsity}


def prune_model(model, amount=0.3, mode="l1_unstructured"):
    """
    Applique l'élagage aux **poids** de toutes les couches `Conv2d` et `Linear`.

    Utilise la réparamétrisation d'élagage de PyTorch (ajoute `weight_orig` et
    `weight_mask`) sans changer la forme du tenseur. Pour intégrer de manière permanente
    les zéros dans les poids stockés, appelez `finalize_pruning(model)` ensuite.

    Parameters
    ----------
    model : nn.Module
        Modèle à élaguer. L'élagage est appliqué **en place** via
        `torch.nn.utils.prune`.
    amount : float, optional (default=0.3)
        Fraction dans [0, 1] à élaguer.
        - Pour l'élagage **non structuré** : fraction des poids de plus petite magnitude
          dans chaque tenseur.
        - Pour l'élagage **structuré (ln)** : fraction des **canaux de sortie**
          (dimension 0) à supprimer en utilisant la norme L2 (n=2).
    mode : {"l1_unstructured", "ln_structured", "global_unstructured"}, optional
        Stratégie d'élagage :
        - `"l1_unstructured"` → `prune.l1_unstructured(..., name="weight", amount=amount)`
        - `"ln_structured"`   → `prune.ln_structured(..., name="weight", amount=amount, n=2, dim=0)`
        - `"global_unstructured"` → `prune.global_unstructured(..., amount=amount)`

    Returns
    -------
    nn.Module
        La même instance de modèle avec la **réparamétrisation** d'élagage appliquée
        (pas encore rendue permanente).
    """

    # Vérifier si amount est dans [0,1]
    if amount < 0 or amount > 1:
        raise ValueError(f"amount doit être dans [0,1], reçu {amount}")

    for _, module in _iter_prunable_modules(model):
        # Vérifier si le module a l'attribut "weight"
        if not hasattr(module, "weight"):
            continue

        # Vérifier si mode est "l1_unstructured"
        if mode == "l1_unstructured":
            # l1_unstructured depuis prune avec module, name("weight"), et amount
            prune.l1_unstructured(module, name="weight", amount=amount)
        # Vérifier si mode est "ln_structured"
        elif mode == "ln_structured":
            # ln_structured depuis prune avec module, name("weight"), amount, n(2), et dim(0)
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
        elif mode == "global_unstructured":
            parameters_to_prune = []
            if hasattr(module, "weight"):
                parameters_to_prune.append((module, "weight"))
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
        else:
            raise ValueError("mode doit être 'l1_unstructured', 'ln_structured' ou 'global_unstructured'")
    return model


def load_model_from_weights(weights_path):
    """
    Charge un checkpoint et reconstruit la tête pour ResNet18.
    """
    # Charger le state_dict
    state = torch.load(weights_path, map_location="cpu")
    # Extraire state_dict si enveloppé
    state = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    # Nettoyer les préfixes model./module. et exclure loss_fn
    state = {
        (k[6:] if k.startswith("model.") else k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
        if not k.startswith("loss_fn.")
    }

    # Créer ResNet18 sans poids ImageNet
    model = tv_models.resnet18(weights=None)
    num_ftrs = model.fc.in_features

    # Reconstruire la tête selon l'architecture sauvegardée
    if "fc.weight" in state:
        # Tête simple Linear
        num_classes = state["fc.weight"].shape[0]
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        # Tête Sequential avec Dropout
        hidden = state["fc.1.weight"].shape[0]
        num_classes = state["fc.4.weight"].shape[0]
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    # Charger les poids
    model.load_state_dict(state, strict=True)
    return model


if __name__ == "__main__":
    # Chemin vers le modèle pré-entraîné
    pretrained_weights = "./models/best_90_resnet18_chest_xray_classifier_weights.pth"
    # Charger le modèle
    model = load_model_from_weights(pretrained_weights)

    # Dataloader de validation
    dm = ChestXRayDataModule(data_dir, batch_size=32)
    dm.setup()
    val_loader = dm.val_dataloader()

    # Créer les statistiques du modèle de référence
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base = sparsity_report(model)
    print("[BASE] global_sparsity:", base["global_sparsity"])

    # Calculer la précision du modèle chargé
    base_acc = helper_utils.compute_accuracy(model, val_loader, device)
    print("[BASE] accuracy:", base_acc)

    # Benchmark sur CPU pour comparaison équitable avec Coursera (probablement CPU)
    cpu_device = torch.device("cpu")
    model_cpu_base = copy.deepcopy(model).to(cpu_device)
    base_time_cpu = helper_utils.bench(model_cpu_base, val_loader, device=cpu_device)
    print("[BASE] time (CPU):", base_time_cpu)

    # Benchmark sur device (MPS)
    base_time = helper_utils.bench(model, val_loader, device=device)
    print("[BASE] time (device):", base_time)

    # On élagage 35% du modèle
    prune_model(model, amount=0.35, mode="l1_unstructured")

    # Statistiques après élagage
    after = sparsity_report(model)
    print("[AFTER PRUNE] global_sparsity:", after["global_sparsity"])

    # Précision après élagage
    after_acc = helper_utils.compute_accuracy(model, val_loader, device)
    print("[AFTER PRUNE] accuracy:", after_acc)

    # Finaliser le pruning pour matérialiser les zéros
    finalize_pruning(model)

    # MPS ne supporte pas les optimisations sparses - tester sur CPU
    model_cpu = model.to(cpu_device)
    pruned_time_cpu = helper_utils.bench(model_cpu, val_loader, device=cpu_device)

    # Aussi tester sur MPS pour référence
    pruned_time = helper_utils.bench(model, val_loader, device=device)

    print("\n=== Comparaison CPU ===")
    print(f"Modèle de base : {base_time_cpu:.4f} secondes par batch")
    print(f"Modèle élagué : {pruned_time_cpu:.4f} secondes par batch")
    print(f"Accélération : {base_time_cpu / pruned_time_cpu:.2f}x")

    print("\n=== Comparaison MPS ===")
    print(f"Modèle de base : {base_time:.4f} secondes par batch")
    print(f"Modèle élagué : {pruned_time:.4f} secondes par batch")
    print(f"Accélération : {base_time / pruned_time:.2f}x")
    print("Note: MPS ne supporte pas les optimisations sparses, d'où le faible speedup.")

    # Sauvegarder le modèle élagué
    torch.save(model.state_dict(), "./models/best_90_resnet18_chest_xray_classifier_weights_pruned.pth")

