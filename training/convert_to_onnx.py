import torch
import numpy as np
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

import utils.helper_utils as helper_utils
from train import ChestXRayDataModule


FP32_PTH_PATH = "../models/best_90_resnet18_chest_xray_classifier_weights_pruned.pth"
ONNX_FP32_PATH = "../models/quantized_fp32_resnet18_chest_xray_weights.onnx"
ONNX_INT8_PATH = "../models/quantized_int8_resnet18_chest_xray_weights.onnx"
DATA_DIR = "./dataset/"


def load_fp32_model(pth_path, num_classes=3):
    """Charge le modèle FP32 pruned avec la bonne architecture."""
    model = helper_utils.QATResNet18(num_classes=num_classes, use_quant_stubs=False)

    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    state = state.get("model_state_dict", state)

    state = {
        (k[6:] if k.startswith("model.") else k[7:] if k.startswith("module.") else k): v
        for k, v in state.items()
        if not k.startswith("loss_fn.")
    }

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def convert_to_onnx_fp32(pth_path, onnx_path, input_size=(1, 3, 224, 224)):
    """Étape 1: Convertit le modèle PyTorch FP32 en ONNX FP32."""
    model = load_fp32_model(pth_path)
    dummy_input = torch.randn(input_size)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,  # opset 17 pour meilleure compatibilité quantification
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"[1/2] Modèle FP32 exporté : {onnx_path}")


class ChestXRayCalibrationReader(CalibrationDataReader):
    """Fournit des données de calibration pour la quantification statique."""

    def __init__(self, data_dir, num_samples=100):
        dm = ChestXRayDataModule(data_dir, batch_size=1)
        dm.setup()

        self.data_iter = iter(dm.val_dataloader())
        self.num_samples = num_samples
        self.count = 0

    def get_next(self):
        if self.count >= self.num_samples:
            return None

        try:
            images, _ = next(self.data_iter)
            self.count += 1
            return {"input": images.numpy().astype(np.float32)}
        except StopIteration:
            return None


def quantize_onnx_int8(fp32_onnx_path, int8_onnx_path, data_dir, num_calibration_samples=100):
    """Étape 2: Quantifie le modèle ONNX FP32 en INT8 avec calibration."""
    print(f"[2/2] Quantification INT8 avec {num_calibration_samples} échantillons de calibration...")

    calibration_reader = ChestXRayCalibrationReader(data_dir, num_samples=num_calibration_samples)

    quantize_static(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        calibration_data_reader=calibration_reader,
        quant_format=ort.quantization.QuantFormat.QDQ,  # Format QDQ pour meilleure compatibilité
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        per_channel=True,  # Quantification par canal pour meilleure précision
    )
    print(f"[2/2] Modèle INT8 quantifié : {int8_onnx_path}")


def get_model_size_mb(path):
    """Retourne la taille du modèle en MB."""
    p = Path(path)
    total = p.stat().st_size
    # Inclure le fichier .data s'il existe (external data)
    data_file = Path(str(path) + ".data")
    if data_file.exists():
        total += data_file.stat().st_size
    return total / (1024 * 1024)


def cleanup_fp32_onnx(fp32_path):
    """Supprime les fichiers ONNX FP32 intermédiaires (.onnx et .onnx.data)."""
    fp32_file = Path(fp32_path)
    data_file = Path(str(fp32_path) + ".data")

    for f in [fp32_file, data_file]:
        if f.exists():
            f.unlink()
            print(f"Nettoyage : {f.name} supprimé")


if __name__ == "__main__":
    # Étape 1: PyTorch FP32 → ONNX FP32
    convert_to_onnx_fp32(FP32_PTH_PATH, ONNX_FP32_PATH)

    # Étape 2: ONNX FP32 → ONNX INT8 (quantification statique avec calibration)
    quantize_onnx_int8(ONNX_FP32_PATH, ONNX_INT8_PATH, DATA_DIR, num_calibration_samples=100)

    # Comparaison des tailles (avant nettoyage)
    fp32_size = get_model_size_mb(ONNX_FP32_PATH)
    int8_size = get_model_size_mb(ONNX_INT8_PATH)

    # Nettoyage des fichiers FP32 intermédiaires
    cleanup_fp32_onnx(ONNX_FP32_PATH)

    print(f"\n=== Résumé ===")
    print(f"Modèle INT8 : {int8_size:.2f} MB (réduction de {((fp32_size - int8_size) / fp32_size * 100):.1f}% vs FP32)")
    print(f"\nConversion terminée ! Utilisez '{ONNX_INT8_PATH}' pour l'inférence.")
