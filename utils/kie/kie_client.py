"""Lightweight client for the Kie.ai Veo3 API."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class KieAIClient:
    """Simple wrapper around Kie.ai's Veo3 endpoints."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("KIE_AI_API_KEY") or os.getenv("KIE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing KIE_AI_API_KEY or KIE_API_KEY environment variable")

        self.base_url = "https://api.kie.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def generate_video(
        self,
        prompt: str,
        *,
        image_urls: Optional[List[str]] = None,
        model: str = "veo3",
        aspect_ratio: str = "9:16",
        seeds: Optional[int] = None,
        watermark: Optional[str] = None,
        callback_url: Optional[str] = None,
        enable_fallback: bool = True,
    ) -> str:
        """Submit a generation request and return the task id."""
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "aspectRatio": aspect_ratio,
            "enableFallback": enable_fallback,
        }

        if image_urls:
            # Kie.ai currently accepts a single guiding frame.
            payload["imageUrls"] = image_urls[:1]
        if seeds is not None:
            payload["seeds"] = seeds
        if watermark:
            payload["watermark"] = watermark
        if callback_url:
            payload["callBackUrl"] = callback_url

        response = requests.post(
            f"{self.base_url}/api/v1/veo/generate",
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        data = response.json()
        if response.ok and data.get("code") == 200:
            return data["data"]["taskId"]
        raise RuntimeError(f"Video generation failed: {data.get('msg', response.text)}")

    def check_status(self, task_id: str) -> Dict[str, Any]:
        """Fetch the latest status record for a task."""
        response = requests.get(
            f"{self.base_url}/api/v1/veo/record-info",
            headers=self.headers,
            params={"taskId": task_id},
            timeout=30,
        )
        data = response.json()
        if response.ok and data.get("code") == 200:
            return data["data"]
        raise RuntimeError(f"Status check failed: {data.get('msg', response.text)}")

    def wait_for_completion(
        self,
        task_id: str,
        *,
        max_wait_time: int = 600,
        poll_interval: int = 15,
    ) -> List[str]:
        """Poll until the task succeeds or errors, returning the result URLs."""
        start = time.time()
        while time.time() - start < max_wait_time:
            status = self.check_status(task_id)
            success_flag = status.get("successFlag", 0)
            if success_flag == 0:
                time.sleep(poll_interval)
                continue
            if success_flag == 1:
                response_payload = status.get("response", {}) or {}
                result_urls = response_payload.get("resultUrls") or status.get("resultUrls")
                if isinstance(result_urls, str):
                    try:
                        result_urls = json.loads(result_urls)
                    except json.JSONDecodeError:
                        result_urls = []
                return result_urls or []
            raise RuntimeError(f"Generation failed with status: {success_flag}")
        raise TimeoutError(f"Timed out waiting for task {task_id}")

    def download_video(self, url: str, output_path: Path) -> Path:
        """Download the generated video to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with open(output_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
        return output_path
