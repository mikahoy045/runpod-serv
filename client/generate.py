import argparse
import base64
import json
import os
import sys
import time
import urllib.request


def _api(endpoint_id, path):
    return f"https://api.runpod.ai/v2/{endpoint_id}{path}"


def _post(url, api_key, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _get(url, api_key):
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _encode_keyframes(batch, base_dir):
    for idea in batch.get("ideas", []):
        for scene in idea.get("scenes", []):
            kf = scene.get("keyframe")
            if kf and not kf.startswith("data:") and not _looks_base64(kf):
                path = kf if os.path.isabs(kf) else os.path.join(base_dir, kf)
                with open(path, "rb") as handle:
                    scene["keyframe"] = base64.b64encode(handle.read()).decode("utf-8")
    return batch


def _looks_base64(value):
    return len(value) > 256 and "/" not in value[:64] and "\\" not in value[:64]


def _save_outputs(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    output = result.get("output", result)
    saved = []
    for idea in output.get("outputs", []):
        if idea.get("status") == "error":
            print(f"  idea {idea.get('idea_id')}: ERROR {idea.get('error')}", file=sys.stderr)
            continue
        for video in idea.get("videos", []):
            name = f"{idea['idea_id']}_{video['filename']}"
            path = os.path.join(out_dir, name)
            with open(path, "wb") as handle:
                handle.write(base64.b64decode(video["data"]))
            saved.append(path)
            print(f"  saved {path}")
    return saved


def run(endpoint_id, api_key, batch, out_dir, poll_seconds):
    response = _post(_api(endpoint_id, "/run"), api_key, {"input": batch})
    job_id = response.get("id")
    print(f"submitted job {job_id}")
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
    while response.get("status") not in terminal:
        time.sleep(poll_seconds)
        response = _get(_api(endpoint_id, f"/status/{job_id}"), api_key)
        print(f"  status: {response.get('status')}")
    if response.get("status") != "COMPLETED":
        print(json.dumps(response, indent=2)[:2000], file=sys.stderr)
        sys.exit(1)
    _save_outputs(response, out_dir)


def main():
    parser = argparse.ArgumentParser(description="Send a batch of ideas/scenes to the LTX-2.3 Runpod endpoint")
    parser.add_argument("--input", required=True, help="Batch JSON file (ideas -> scenes)")
    parser.add_argument("--out-dir", default="output", help="Where to save returned movies")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    args = parser.parse_args()
    if not args.endpoint_id or not args.api_key:
        parser.error("set --endpoint-id/--api-key or RUNPOD_ENDPOINT_ID/RUNPOD_API_KEY")

    with open(args.input, "r", encoding="utf-8") as handle:
        batch = json.load(handle)
    batch = batch.get("input", batch)
    batch = _encode_keyframes(batch, os.path.dirname(os.path.abspath(args.input)))
    run(args.endpoint_id, args.api_key, batch, args.out_dir, args.poll_seconds)


if __name__ == "__main__":
    main()
