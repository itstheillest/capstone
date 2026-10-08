# PixiFAI — Automated Image Analysis & Forensic Pipeline

PixiFAI is a production-ready, containerized image analysis and verification pipeline. Built on Django 6.0, Celery, Redis, and PyTorch/YOLOv8, it provides asynchronous image manipulation detection, metadata auditing, face detection, and object tagging via a JWT-secured REST API documented with Swagger UI.

---

## Tech Stack

* **Core Framework**: Django 6.0, Django REST Framework (DRF)
* **Async Processing & Broker**: Celery, Redis
* **Database**: PostgreSQL
* **Machine Learning & Vision**: PyTorch (Custom Forensic CNN), Ultralytics YOLOv8, OpenCV, Pillow
* **Authentication**: `djangorestframework-simplejwt`
* **API Documentation**: `drf-spectacular` (OpenAPI 3.0 / Swagger UI)
* **Cross-Origin Support**: `django-cors-headers`
* **Infrastructure**: Docker & Docker Compose (WSL2 optimized)

---

## System Architecture & Pipeline Lifecycle

```
                     +-------------------------------------------------+
                     |                 Docker Engine                   |
                     +------------------------+------------------------+
                                              |
                      Port 8000 (HTTP)        |        Port 6379 (Internal)
               +------------------------------+----------------------------+
               |                                                           |
               v                                                           v
     +--------------------+    Shared Volume    +--------------------+  +------------+
     |       `web`        | <=================> |      `celery`      |  |  `redis`   |
     |  (Django 6.0 API)  |    `/app/media`     |  (Async Pipeline)  |  |  (Broker)  |
     +---------+----------+                     +---------+----------+  +------------+
               |                                          |
               +--------------------+---------------------+
                                    |  Postgres Port 5432 (Internal)
                                    v
                            +---------------+
                            |     `db`      |
                            | (PostgreSQL)  |
                            +---------------+
```

### Pipeline Workflow & State Lifecycle

1. **Submission**: Client uploads an image via `POST /api/submissions/` (`multipart/form-data`).
2. **Persistence & Queueing**: Django saves the raw file to `/app/media/submissions/`, sets status to `PENDING`, and triggers a Celery background task (`process_submission`).
3. **Asynchronous Execution**:
   * **EXIF Stage**: Extracts embedded camera metadata, GPS, and creation dates.
   * **Face Detection Stage**: Scans for human face boundaries using OpenCV.
   * **Forensic CNN Stage**: Computes probability scores (0.0 to 1.0) using PyTorch to detect AI generation or digital manipulation.
   * **Tagging Stage**: Identifies visual entities and object categories via YOLOv8.
   * **Fact-Check Stage**: Cross-references reverse image match indices and context references.
4. **Completion**: Results are saved to a `DetectionReport` record linked to the submission, and status is updated to `COMPLETED` (or `FAILED`).

---

## Memory & WSL2 Runtime Optimizations

To prevent `Errno 12` (Out of Memory / Cannot allocate memory) crashes on resource-constrained development environments:

* **Lazy Model Initialization**: ML models (PyTorch CNN and YOLOv8) are loaded strictly on-demand inside service getters during inference—never during Docker container boot or Django initialization.
* **Worker Concurrency Cap**: Celery runs with `--concurrency=2` to limit parallel RAM allocations.
* **CPU Thread Regulation**: `OMP_NUM_THREADS=1` is injected across services to prevent unbounded OpenMP/PyTorch thread spawning.
* **Stateless Server Execution**: Django runs with `--noreload` inside `docker-compose.yml` to prevent auto-reloader process duplicates.

---

## Quick Start (Docker Environment)

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed with WSL2 backend enabled.

### 1. Environment Setup

Create a `.env` file in the root project folder:


### 2. Boot Services & Database

Run the following commands in your terminal:

```powershell
# 1. Build and launch all containers in detached mode
docker compose up -d

# 2. Execute database migrations
docker compose exec web python manage.py migrate

# 3. Create an administrative account
docker compose exec web python manage.py createsuperuser
```

### 3. Verify Container Health

```powershell
docker compose ps
```
*Ensure `web`, `celery`, `db`, and `redis` containers display `Up` / `running` status.*

---

## Authentication & API Documentation

* **Interactive Swagger UI**: `http://localhost:8000/api/docs/`
* **OpenAPI Raw Schema**: `http://localhost:8000/api/schema/`

### Authorization Workflow

1. Send `POST /api/token/` with body `{ "username": "...", "password": "..." }`.
2. Copy the returned `"access"` token.
3. In Swagger UI, click **Authorize** at the top right, enter `Bearer <your_access_token>`, and click **Authorize**.

---

## Frontend Developer Onboarding Guide

### 1. Auto-Generating TypeScript Interfaces

Sync frontend TypeScript interfaces directly with backend Django models by running:

```bash
npx openapi-typescript http://localhost:8000/api/schema/ -o src/types/api.ts
```

### 2. Async Polling Pattern

Since forensic evaluation runs asynchronously, frontend applications must poll for status updates:

1. Send `POST /api/submissions/` (`multipart/form-data`) containing the image payload.
2. Store the returned submission `id` (UUID).
3. Poll `GET /api/submissions/{id}/` every **2 to 3 seconds**.
4. Stop polling when `status` shifts to `"COMPLETED"` or `"FAILED"`.

### 3. Key Endpoints Reference

| HTTP Method | Endpoint | Description | Content-Type |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/token/` | Obtain JWT access/refresh token pair | `application/json` |
| `POST` | `/api/token/refresh/` | Refresh expired access token | `application/json` |
| `GET` | `/api/submissions/` | List all user submissions | N/A |
| `POST` | `/api/submissions/` | Upload image for analysis | `multipart/form-data` |
| `GET` | `/api/submissions/{id}/` | Fetch status and full report | N/A |

### 4. Client File Constraints
* **Allowed Extensions**: `.jpg`, `.jpeg`, `.png`, `.webp`
* **Maximum File Size**: `10 MB`

---

## Repository Structure

```text
pixifai/
├── detection/               # Django application (Pipeline & API)
│   ├── models.py            # ImageSubmission and DetectionReport models
│   ├── views.py             # ViewSets with MultiPartParser support
│   ├── serializers.py       # DRF Serializers with OpenAPI schema hooks
│   ├── pipeline.py          # Master async pipeline orchestration
│   └── services/            # Micro-services
│       ├── exif_service.py      # EXIF metadata extraction
│       ├── face_service.py      # OpenCV face detection
│       ├── forensic_service.py  # PyTorch forensic model lazy-loading & inference
│       ├── vision_service.py    # YOLOv8 object detection & tagging
│       └── factcheck_service.py # Reference verification logic
├── model_weights/           # Local storage for .pt / .pth model weights
├── pixifai/                 # Project configuration directory
│   ├── settings.py          # REST framework, Celery, & CORS configurations
│   ├── urls.py              # Central routing & Swagger UI views
│   └── celery.py            # Celery instance configuration
├── docker-compose.yml       # Orchestration file
├── Dockerfile               # Python environment setup
└── requirements.txt         # Core dependencies
```