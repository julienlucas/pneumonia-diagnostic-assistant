import copy
import torch
import torch.nn as nn
from torchvision import models as tv_models
from train import ChestXRayDataModule, data_dir
import utils.helper_utils as helper_utils


def quantize_dynamic_linear(model):
    """
    Quantifie dynamiquement toutes les couches Linear en INT8.
    Retourne un nouveau modèle en mode eval.
    """
    # Créer une copie profonde du modèle et le mettre en mode eval
    model_fp32 = copy.deepcopy(model).eval()

    # Assurer un moteur approprié sur CPU (x86). Si indisponible, cette ligne est inoffensive.
    # Vérifier si torch.backends a quantized
    has_quantized = hasattr(torch.backends, "quantized")
    # Vérifier si torch.backends.quantized a engine
    has_engine = hasattr(torch.backends.quantized, "engine") if has_quantized else False
    if has_quantized and has_engine:
        try:
            # qnnpack pour MPS et fbgemm pour CPU
            torch.backends.quantized.engine = "qnnpack"
        except Exception:
            # garder ce que le runtime supporte
            pass

    # Quantifier uniquement les couches Linear en INT8
    # Utiliser quantize_dynamic depuis ao.quantization dans torch pour quantifier model_fp32 en INT8
    quantized = torch.ao.quantization.quantize_dynamic(
        model_fp32,  # Le modèle à quantifier
        {nn.Linear},  # Les couches à quantifier (uniquement les couches Linear)
        dtype=torch.qint8,  # Le dtype pour quantifier en qint8
    )

    # Mettre le modèle quantifié en mode eval
    quantized.eval()
    # Retourner le modèle quantifié
    return quantized


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
    pretrained_weights = "./models/best_90_resnet18_chest_xray_classifier_weights_pruned.pth"
    # Charger le modèle
    model = load_model_from_weights(pretrained_weights)

    # Dataloader de validation
    dm = ChestXRayDataModule(data_dir, batch_size=32)
    dm.setup()
    val_loader = dm.val_dataloader()

    # Évaluation et benchmarks CPU
    # La quantification ne fonctionne que sur CPU
    cpu_device = torch.device("cpu")
    model_cpu = model.to(cpu_device).eval()

    # Calculer la précision du modèle de base
    base_acc = helper_utils.compute_accuracy(model_cpu, val_loader, device=cpu_device)
    print(f"[BASE] accuracy: {base_acc:.4f}")

    # Benchmark FP32 sur CPU
    base_time_cpu = helper_utils.bench(model_cpu, val_loader, device=cpu_device)
    print("[BASE] time (CPU):", base_time_cpu)

    # Quantifier le modèle (uniquement sur CPU, la quantification ne fonctionne pas sur MPS)
    qmodel = quantize_dynamic_linear(model_cpu)

    # Évaluer le modèle quantifié
    qacc = helper_utils.compute_accuracy(qmodel, val_loader, device=cpu_device)
    print(f"[AFTER QUANTIZE] accuracy: {qacc:.4f}")

    # Benchmark INT8 sur CPU (la quantification ne fonctionne que sur CPU)
    t_int8_cpu = helper_utils.bench(qmodel, val_loader, device=cpu_device)
    print("\n=== Comparaison CPU ===")
    print(f"Modèle FP32 : {base_time_cpu*1e3:.2f} ms par batch")
    print(f"Modèle INT8 : {t_int8_cpu*1e3:.2f} ms par batch")
    print(f"Amélioration : {((base_time_cpu - t_int8_cpu)/base_time_cpu)*100:.1f}%")
    print(f"Accélération : {base_time_cpu / t_int8_cpu:.2f}x")

    # Sauvegarder le modèle quantifié
    torch.save(qmodel.state_dict(), "./models/best_90_resnet18_chest_xray_classifier_weights_pruned_global_quantized.pth")