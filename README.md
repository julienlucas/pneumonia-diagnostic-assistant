# Pneumonia Diagnostic Assistant

Projet démo pour détecter la pneumonie à partir de radiographies (3 classes : Normal, Pneumonie bactérienne, Pneumonie virale).

## Prérequis
- Python 3.11+
- Node.js 20+ et pnpm
- uv

## Installation

```bash
uv sync
```

```bash
cd frontend
pnpm install
```

## Lancer en local

Backend :
```bash
python manage.py runserver 0.0.0.0:8000
```

Frontend :
```bash
cd frontend
pnpm run dev
```

## Conversion ONNX
```bash
python convert_to_onnx.py
```

## Notes
- Le modèle ONNX se trouve dans `backend/model/`.
- Les images d’exemple sont dans `static/`.
