import hashlib
from pathlib import Path


class DiskCache:
    """Stores raw HTML responses on disk to avoid re-fetching."""

    def __init__(self, cache_dir: str = "data/raw"):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, url: str) -> Path:
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self._dir / f"{key}.html"

    def get(self, url: str) -> str | None:
        p = self._path(url)
        if p.exists():
            return p.read_text(encoding="utf-8")
        return None

    def set(self, url: str, content: str) -> None:
        self._path(url).write_text(content, encoding="utf-8")

    def has(self, url: str) -> bool:
        return self._path(url).exists()
