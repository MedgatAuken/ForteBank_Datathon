# -*- coding: utf-8 -*-
"""
FastAPI-сервис.

Пути и заголовок заданы ТЗ дословно:
    POST /api/v1/analyze        поле file, заголовок X-API-Key  -> {"job_id": ...}
    GET  /api/v1/jobs/{job_id}  заголовок X-API-Key             -> статус + результат

Запуск:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Optional, Union

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.jobs import store
from app.schemas import AnalyzeResponse, JobResponse

app = FastAPI(
    title="ForteBank — контроль кассовой дисциплины",
    description="Локальный анализ видеозаписей работы кассира. Без облачных AI-сервисов.",
    version="0.1.0",
)

STATIC_DIR = config.BASE_DIR / "static"
# Раздача статики интерфейса (стили, скрипты, изображения).
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def require_key(x_api_key: Optional[str]) -> None:
    if not x_api_key or x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="неверный или отсутствующий X-API-Key")


@app.get("/health")
def health() -> dict:
    """Проверка живости. Без ключа — чтобы жюри могло убедиться, что сервис поднят."""
    return {
        "status": "ok",
        "analyzer": store.analyzer_name,
        "max_processing_sec": config.MAX_PROCESSING_SEC,
    }


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> AnalyzeResponse:
    require_key(x_api_key)

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in config.ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"ожидается один из {sorted(config.ALLOWED_SUFFIXES)}, получено '{suffix}'",
        )

    # Пишем потоково: 29 МБ в память класть незачем, а жюри может слать параллельно.
    dest = config.UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"файл больше {config.MAX_UPLOAD_MB:g} МБ",
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="пустой файл")

    # Возвращаем job_id немедленно — обработка уходит в фон.
    return AnalyzeResponse(job_id=store.submit(dest))


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(
    job_id: str,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> JobResponse:
    require_key(x_api_key)
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job_id не найден")
    return job


@app.get("/", response_model=None)
def index() -> Union[FileResponse, JSONResponse]:
    page = STATIC_DIR / "index.html"
    if not page.exists():
        return JSONResponse({"detail": "UI не собран, см. app/static/index.html"}, 404)
    return FileResponse(page)
