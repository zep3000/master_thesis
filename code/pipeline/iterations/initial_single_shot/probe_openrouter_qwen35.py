#!/usr/bin/env python3
"""Probe OpenRouter Qwen3.5-9B endpoint metadata and lightweight provider behavior."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

from run_single_shot_first20 import (
    DEFAULT_IMAGE_DIR,
    DEFAULT_KEY_FILE,
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    build_messages,
    build_openrouter_client,
    image_to_data_url,
    iso_now,
    load_api_key,
    read_json,
    resolve_path,
    select_manifest_images,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "output" / "provider_probe.json"
DEFAULT_PROVIDER_TAGS = ["deepinfra/bf16", "parasail/bf16", "venice/fp8", "siliconflow/fp8"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider-tag", action="append", dest="provider_tags")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = load_api_key(resolve_path(args.api_key_file))
    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provider_tags = args.provider_tags or DEFAULT_PROVIDER_TAGS
    endpoint_metadata = fetch_endpoint_metadata(args.base_url, args.model, api_key)
    smoke_results = []
    if not args.skip_smoke:
        manifest = read_json(resolve_path(args.manifest))
        first_item = select_manifest_images(manifest, 0, 1)[0]
        image_path = resolve_path(args.image_dir) / first_item["filename"]
        data_url, payload_info = image_to_data_url(image_path, max_image_mb=0)
        prompt = (
            "Return only JSON: {\"ok\":true,\"visible_faces_estimate\":integer,\"note\":\"short\"}. "
            "Estimate visible face depictions on this historical advertisement page."
        )
        client = build_openrouter_client(
            api_key=api_key,
            base_url=args.base_url,
            app_name="economist-ad-face-qwen-provider-probe",
            http_referer=None,
            timeout=args.request_timeout_seconds,
        )
        for provider_tag in provider_tags:
            smoke_results.append(smoke_provider(client, args.model, provider_tag, prompt, data_url, payload_info))
    payload = {
        "schema_version": "qwen_iteration_provider_probe_v1",
        "generated_at": iso_now(),
        "model": args.model,
        "endpoint_metadata": endpoint_metadata,
        "smoke_results": smoke_results,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summarize(payload), indent=2, ensure_ascii=False))
    return 0


def fetch_endpoint_metadata(base_url: str, model: str, api_key: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/models/{model}/endpoints"
    session = requests.Session()
    session.trust_env = False
    response = session.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
    response.raise_for_status()
    return response.json()


def smoke_provider(client: Any, model: str, provider_tag: str, prompt: str, data_url: str, payload_info: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    result: dict[str, Any] = {
        "provider_tag": provider_tag,
        "payload_info": payload_info,
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(prompt, data_url),
            max_tokens=200,
            temperature=0,
            response_format={"type": "json_object"},
            extra_body={
                "provider": {
                    "order": [provider_tag],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
                "reasoning": {"effort": "none", "exclude": True},
            },
        )
        message = response.choices[0].message.content
        result["ok"] = True
        result["response_text"] = message
        result["model"] = getattr(response, "model", None)
        result["usage"] = getattr(response, "usage", None).model_dump() if getattr(response, "usage", None) else None
    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_seconds"] = round(time.time() - started, 3)
    return result


def summarize(payload: dict[str, Any]) -> dict[str, Any]:
    endpoints = payload.get("endpoint_metadata", {}).get("data", {}).get("endpoints", [])
    endpoint_summary = [
        {
            "tag": endpoint.get("tag"),
            "provider_name": endpoint.get("provider_name"),
            "quantization": endpoint.get("quantization"),
            "context_length": endpoint.get("context_length"),
            "max_completion_tokens": endpoint.get("max_completion_tokens"),
            "status": endpoint.get("status"),
            "uptime_last_5m": endpoint.get("uptime_last_5m"),
            "latency_p50_30m": endpoint.get("latency_last_30m", {}).get("p50") if isinstance(endpoint.get("latency_last_30m"), dict) else None,
            "throughput_p50_30m": endpoint.get("throughput_last_30m", {}).get("p50") if isinstance(endpoint.get("throughput_last_30m"), dict) else None,
            "supported_parameters": endpoint.get("supported_parameters"),
        }
        for endpoint in endpoints
    ]
    return {
        "model": payload.get("model"),
        "endpoints": endpoint_summary,
        "smoke": [
            {
                "provider_tag": item.get("provider_tag"),
                "ok": item.get("ok"),
                "elapsed_seconds": item.get("elapsed_seconds"),
                "error": item.get("error"),
                "response_text": item.get("response_text"),
            }
            for item in payload.get("smoke_results", [])
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
