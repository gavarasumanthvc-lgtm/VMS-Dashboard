"""HuggingFace router VLM client."""

import time
import requests

from config import (
    API_RETRY_ON_503_DELAY_SECONDS,
    API_TIMEOUT_SECONDS,
    HF_API_URL,
    HF_MODEL,
    STEP_SCORE_LIMITS,
)


def query_vlm_multi(frames_b64, prompt, step_name, headers, log=None):
    """Send multiple frames for one SOP step; the model sees all frames together."""
    if log:
        log(f"Calling API for: {step_name} ({len(frames_b64)} frames), model={HF_MODEL}")

    try:
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            for b64 in frames_b64
        ]
        max_score = STEP_SCORE_LIMITS[step_name]
        content.append({
            "type": "text",
            "text": (
                f"{prompt}\n\nUse only visible evidence. End with exactly `SCORE: n/{max_score}`, "
                "where n is an integer from 0 to the maximum."
            ),
        })

        payload = {
            "model": HF_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 250,
            "temperature": 0.1,
        }

        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT_SECONDS)
        if log:
            log(f"Response status: {response.status_code}")

        if response.status_code == 503:
            if log:
                log("Model loading — retrying once...", "WARN")
            time.sleep(API_RETRY_ON_503_DELAY_SECONDS)
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT_SECONDS)
            if log:
                log(f"Retry status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, dict) and "choices" in result:
                text = result["choices"][0]["message"]["content"]
                if log:
                    log(f"Answer: {text[:150]}")
                return text
            return str(result)

        error = response.text[:300]
        if log:
            log(f"API ERROR {response.status_code}: {error}", "ERROR")
        return f"API error {response.status_code}: {error}"

    except requests.exceptions.Timeout:
        if log:
            log(f"Timeout after {API_TIMEOUT_SECONDS}s", "ERROR")
        return "Error: Request timed out"
    except Exception as e:
        if log:
            log(f"Exception: {str(e)}", "ERROR")
        return f"Error: {str(e)}"
