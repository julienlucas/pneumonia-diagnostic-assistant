# Pneumonia Diagnostic Assistant

App **fullstack** pour détecter la pneumonie à partir de radiographies.
Classes : **Normal**, **Pneumonie bactérienne**, **Pneumonie virale**.

## 🔍 Principe

Ce projet repose sur un **transfer learning** d’un modèle ResNet18 :

- **Modèle de base** : ResNet18 (ImageNet)
- **Méthode** : fine‑tuning du classifieur
- **Dataset** : environ 450 radiographies (3 classes)
- **Objectif** : modèle léger, rapide et exploitable en prod

## 🧠 Inference

L’inférence se fait via **ONNX Runtime** pour réduire la latence.

Le script `backend/inference.py` génère aussi une **heatmap Grad‑CAM**.

## 📦 Installation

### Backend
```bash
uv sync
```

### Frontend
```bash
cd frontend
pnpm install
```

## ▶️ Lancer en local

### Backend
```bash
python manage.py runserver 0.0.0.0:8000
```

### Frontend
```bash
cd frontend
pnpm run dev
```

## 🧪 Conversion ONNX

```bash
python convert_to_onnx.py
```

## 📁 Arborescence utile

- Modèle ONNX : `backend/model/`
- Images d’exemple : `static/`

## 📄 Notes

Ce projet est une **démo technique**. Les performances varient selon le jeu de données
et ne remplacent pas une validation clinique.
