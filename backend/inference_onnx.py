import os
import cv2
import argparse
import torch
import torch.nn as nn
import torchvision.models as tv_models
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

PTH_MODEL_PATH = "./backend/model/resnet18_chest_xray_classifier_weights.pth"
ONNX_MODEL_PATH = "./backend/model/resnet18_chest_xray_classifier_weights.onnx"

CLASS_NAMES = ["BACTERIAL_PNEUMONIA", "NORMAL", "VIRAL_PNEUMONIA"]
CLASS_LABELS = {
    "BACTERIAL_PNEUMONIA": "Pneumonie bactérienne",
    "NORMAL": "Normal",
    "VIRAL_PNEUMONIA": "Pneumonie virale",
}
LABEL_COLORS = {
    "NORMAL": (255, 255, 255),
    "VIRAL_PNEUMONIA": (255, 0, 0),
    "BACTERIAL_PNEUMONIA": (255, 165, 0),
}

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
]

_onnx_session = None
_pytorch_model = None

def load_fixed_font(width):
    font_size = max(18, int(width * 0.02734))
    candidates = list(FONT_CANDIDATES)
    pil_fonts_dir = os.path.join(os.path.dirname(ImageFont.__file__), "fonts")
    candidates.append(os.path.join(pil_fonts_dir, "DejaVuSans.ttf"))
    for path in candidates:
        try:
            return ImageFont.truetype(path, font_size), True, font_size
        except Exception:
            continue
    return ImageFont.load_default(), False, font_size

def draw_label(pil_image, text, rgb_color):
    draw = ImageDraw.Draw(pil_image)
    width, _height = pil_image.size
    font, is_truetype, font_size = load_fixed_font(width)
    if is_truetype:
        draw.text((12, 12), text, fill=rgb_color, font=font)
        return
    text_w, text_h = draw.textbbox((0, 0), text, font=font)[2:]
    scale = max(1, int(round(font_size / max(1, text_h))))
    render_w = max(1, int(text_w * scale))
    render_h = max(1, int(text_h * scale))
    mask = Image.new("L", (text_w, text_h), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.text((0, 0), text, fill=255, font=font)
    if scale != 1:
        mask = mask.resize((render_w, render_h), resample=Image.NEAREST)
    color_img = Image.new("RGBA", (render_w, render_h), rgb_color + (255,))
    pil_image.paste(color_img, (12, 12), mask)

def get_pytorch_model():
    global _pytorch_model
    if _pytorch_model is None:
        _pytorch_model = load_resnet18_model(PTH_MODEL_PATH, num_classes=len(CLASS_NAMES))
        _pytorch_model.eval()
    return _pytorch_model

def get_onnx_session():
    global _onnx_session
    if _onnx_session is None:
        _onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
    return _onnx_session

def load_resnet18_model(weights_path, num_classes=None):
    model = tv_models.resnet18(weights=None)
    if num_classes is not None:
        num_features = model.fc.in_features
        model.fc = nn.Linear(in_features=num_features, out_features=num_classes)
    state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    return model

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._on_forward)

    def _on_forward(self, _module, _inputs, output):
        self.activations = output.detach()
        def _on_backward(grad):
            self.gradients = grad.detach()
        output.register_hook(_on_backward)

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        self.model.zero_grad(set_to_none=True)
        output = self.model(x)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1).item())
        score = output[:, class_idx].sum()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=False)
        cam = cam.relu()[0]
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        return cam.detach().cpu().numpy(), class_idx

def predict_with_saliency(pil_image):
    session = get_onnx_session()
    model = get_pytorch_model()

    w, h = pil_image.size

    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_resized = pil_image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = (img_array - imagenet_mean) / imagenet_std
    img_array = img_array.transpose(2, 0, 1)

    outputs = session.run(None, {"input": img_array[np.newaxis, ...]})
    logits = outputs[0][0] if outputs[0].ndim == 2 else outputs[0]
    probs = torch.softmax(torch.from_numpy(logits), dim=0)
    pred_idx = int(probs.argmax().item())

    pred_class = CLASS_NAMES[pred_idx]
    pred_label = CLASS_LABELS.get(pred_class, pred_class)
    text_color = LABEL_COLORS.get(pred_class, (255, 255, 255))
    conf = float(probs[pred_idx].item())

    bacterial_conf = float(probs[0].item())
    normal_conf = float(probs[1].item())
    viral_conf = float(probs[2].item())

    input_tensor = torch.from_numpy(img_array).unsqueeze(0)
    grad_cam = GradCAM(model, model.layer4[-1].conv2)
    heatmap_small, _ = grad_cam(input_tensor, pred_idx)

    img_display = np.array(pil_image)
    heatmap_resized = cv2.resize(heatmap_small, (w, h))

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET,
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(
        (img_display).astype(np.uint8),
        0.6,
        heatmap_color,
        0.4,
        0,
    )

    result_img = Image.fromarray(superimposed)
    draw_label(result_img, f"{pred_label} ({conf * 100:.1f}%)", text_color)

    return result_img, pred_label, conf, normal_conf, bacterial_conf, viral_conf

def predict_with_gradcam(pil_image):
    return predict_with_saliency(pil_image)

def predict_and_draw_saliency(image_path):
    session = get_onnx_session()
    model = get_pytorch_model()
    pil_image = Image.open(image_path).convert("RGB")
    w, h = pil_image.size

    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    img_resized = pil_image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = (img_array - imagenet_mean) / imagenet_std
    img_array = img_array.transpose(2, 0, 1)

    outputs = session.run(None, {"input": img_array[np.newaxis, ...]})
    logits = outputs[0][0] if outputs[0].ndim == 2 else outputs[0]
    probs = torch.softmax(torch.from_numpy(logits), dim=0)
    pred_idx = int(probs.argmax().item())

    input_tensor = torch.from_numpy(img_array).unsqueeze(0)
    grad_cam = GradCAM(model, model.layer4[-1].conv2)
    heatmap_small, _ = grad_cam(input_tensor, pred_idx)

    img_display = np.array(pil_image)
    heatmap_resized = cv2.resize(heatmap_small, (w, h))

    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        cv2.COLORMAP_JET,
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(
        (img_display).astype(np.uint8),
        0.6,
        heatmap_color,
        0.4,
        0,
    )
    pred_class = CLASS_NAMES[pred_idx]
    pred_label = CLASS_LABELS.get(pred_class, pred_class)
    text_color = LABEL_COLORS.get(pred_class, (255, 255, 255))
    result_img = Image.fromarray(superimposed)
    draw_label(result_img, f"{pred_label} ({float(probs[pred_idx].item()) * 100:.1f}%)", text_color)

    return result_img, CLASS_NAMES[pred_idx], float(probs[pred_idx].item())

def main():
    parser = argparse.ArgumentParser(description="Prédit la classe d'une radiographie")
    parser.add_argument("image_name", help="Nom de l'image dans static/<classe>/")
    args = parser.parse_args()

    static_dir = "./static"
    image_path = None
    if os.path.isdir(static_dir):
        for entry in os.listdir(static_dir):
            candidate = os.path.join(static_dir, entry, args.image_name)
            if os.path.isfile(candidate):
                image_path = candidate
                break

    if image_path is None:
        raise FileNotFoundError(
            f"Image '{args.image_name}' introuvable dans static/*/"
        )

    predict_and_draw_saliency(image_path)

if __name__ == "__main__":
    main()