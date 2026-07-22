# PixifAI

Deepfake / manipulated-image detection tool for [agency], built with Django,
DRF, and a computer-vision/ML pipeline (OpenCV, ExifTool, face_recognition,
DeepFace, YOLOv8, CNN/ViT classifier).

## Local setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

Visit `http://127.0.0.1:8000/admin/` to browse the data model.

## Apps

- `accounts` — custom User model (roles: analyst, supervisor, admin, viewer)
- `detection` — core pipeline: submissions, EXIF, faces, forensic analysis,
  reverse image matches, fact-checks, final report
- `legalmap` — maps detection outcomes to specific Philippine legal provisions
- `audit` — append-only chain-of-custody log

## Environment variables

Copy `.env.example` to `.env` and fill in real values before running with
Postgres or calling external APIs.
