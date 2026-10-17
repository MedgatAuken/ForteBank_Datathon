# -*- coding: utf-8 -*-
"""
Очередь задач и watchdog.

Главное требование ТЗ: жюри должно получить ответ быстрее 5 минут, иначе видео
не засчитано. Поэтому здесь ДВА уровня защиты:

  1. Кооперативный Deadline — пайплайн сам проверяет его и выходит досрочно.
  2. Жёсткий watchdog — если пайплайн всё-таки завис и не опрашивает Deadline,
     таймер принудительно переводит задачу в done с тем, что успело попасть в sink.
     Повисший поток продолжит работать, но его результат будет отброшен.

Падать в failed мы не хотим почти никогда: частичный ответ лучше, чем никакого.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app import config
from app.pipeline import Deadline, load_analyzer
from app.schemas import JobResponse, Violation

WATCHDOG_GRACE_SEC = 10.0


@dataclass
class Job:
    job_id: str
    video_path: Path
    status: str = "processing"
    verdict: Optional[str] = None
    violations: list[Violation] = field(default_factory=list)
    error: Optional[str] = None
    started_at: float = field(default_factory=time.monotonic)
    processing_sec: Optional[float] = None
    truncated: bool = False
    _finalized: bool = False


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=config.WORKERS,
                                        thread_name_prefix="analyze")
        self._analyzer = load_analyzer()

    @property
    def analyzer_name(self) -> str:
        return self._analyzer.name

    def submit(self, video_path: Path) -> str:
        job_id = uuid.uuid4().hex[:8]
        job = Job(job_id=job_id, video_path=video_path)
        with self._lock:
            self._jobs[job_id] = job
        self._pool.submit(self._run, job)
        timer = threading.Timer(
            config.MAX_PROCESSING_SEC + WATCHDOG_GRACE_SEC, self._force_finish, (job_id,)
        )
        timer.daemon = True
        timer.start()
        return job_id

    def get(self, job_id: str) -> Optional[JobResponse]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return JobResponse(
                job_id=job.job_id,
                status=job.status,          # type: ignore[arg-type]
                verdict=job.verdict,        # type: ignore[arg-type]
                violations=list(job.violations),
                error=job.error,
                processing_sec=job.processing_sec,
                truncated=job.truncated,
            )

    # ------------------------------------------------------------- внутреннее
    def _run(self, job: Job) -> None:
        deadline = Deadline(config.MAX_PROCESSING_SEC)
        sink: list[Violation] = []
        # sink виден снаружи сразу — watchdog заберёт из него частичный результат
        job.violations = sink
        try:
            verdict = self._analyzer.analyze(job.video_path, deadline, sink)
            self._finish(job, verdict=verdict, truncated=False)
        except TimeoutError:
            self._finish(job, verdict=None, truncated=True)
        except Exception:
            traceback.print_exc()
            with self._lock:
                if job._finalized:
                    return
                job._finalized = True
                job.status = "failed"
                job.error = "внутренняя ошибка обработки"
                job.processing_sec = round(time.monotonic() - job.started_at, 2)

    def _finish(self, job: Job, verdict: Optional[str], truncated: bool) -> None:
        with self._lock:
            if job._finalized:
                return
            job._finalized = True
            # вердикт выводим из найденного, если пайплайн не успел его вернуть
            job.verdict = verdict or ("violation" if job.violations else "no_violation")
            job.status = "done"
            job.truncated = truncated
            job.processing_sec = round(time.monotonic() - job.started_at, 2)

    def _force_finish(self, job_id: str) -> None:
        """Сработал жёсткий watchdog: пайплайн не уложился и не отреагировал на Deadline."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job._finalized:
                return
            job._finalized = True
            job.verdict = "violation" if job.violations else "no_violation"
            job.status = "done"
            job.truncated = True
            job.processing_sec = round(time.monotonic() - job.started_at, 2)
        print(f"[watchdog] job {job_id} принудительно завершён по таймауту")


store = JobStore()
