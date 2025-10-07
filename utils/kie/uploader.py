"""Helpers for pushing still images to temporary hosting for Kie.ai."""
from __future__ import annotations

import os
from typing import Literal

import requests

ServiceName = Literal["catbox", "fileio"]


def upload_image(image_path: str, service: ServiceName = "catbox") -> str:
    """Upload ``image_path`` and return a public URL."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if service == "catbox":
        return _upload_catbox(image_path)
    if service == "fileio":
        return _upload_fileio(image_path)
    raise ValueError(f"Unsupported upload service: {service}")


def _upload_catbox(image_path: str) -> str:
    with open(image_path, "rb") as handle:
        response = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload"},
            files={"fileToUpload": handle},
            timeout=120,
        )
    response.raise_for_status()
    url = response.text.strip()
    if not url.startswith("http"):
        raise RuntimeError(f"Catbox upload failed: {response.text}")
    return url


def _upload_fileio(image_path: str) -> str:
    with open(image_path, "rb") as handle:
        response = requests.post(
            "https://file.io",
            files={"file": handle},
            data={"expires": "1d"},
            timeout=120,
        )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise RuntimeError(f"file.io upload failed: {payload}")
    return payload["link"]
