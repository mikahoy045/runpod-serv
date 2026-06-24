import base64
import json
import os
import struct
import subprocess
import time
import urllib.request
import uuid
import zlib

import requests
import runpod

import config
from scene_workflow import build_scene_workflow, resolve_seconds

HOST = config.COMFY_HOST
_base = None


def _reconcile(workflow, object_info):
    for node in workflow.values():
        spec = object_info.get(node["class_type"], {}).get("input", {})
        allowed = set(spec.get("required", {})) | set(spec.get("optional", {}))
        if allowed:
            node["inputs"] = {k: v for k, v in node["inputs"].items() if (k in allowed) or isinstance(v, (list, dict))}
    return workflow


def _wait_for_comfy():
    for _ in range(600):
        try:
            if requests.get(f"http://{HOST}/system_stats", timeout=5).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("ComfyUI did not become ready")


def _load_base():
    global _base
    if _base is None:
        _wait_for_comfy()
        object_info = requests.get(f"http://{HOST}/object_info", timeout=120).json()
        with open(config.WORKFLOW_PATH, "r", encoding="utf-8") as handle:
            _base = _reconcile(json.load(handle), object_info)
    return _base


def _wait_for_models():
    marker = config.MODELS_READY_FILE
    if not marker:
        return
    deadline = time.time() + config.MODELS_READY_TIMEOUT
    while time.time() < deadline:
        if os.path.exists(marker):
            return
        time.sleep(2)
    raise RuntimeError("models not ready: background download did not complete within timeout")


def _png(width, height, rgb=(120, 110, 90)):
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _upload_image(name, blob):
    requests.post(
        f"http://{HOST}/upload/image",
        files={"image": (name, blob, "image/png"), "overwrite": (None, "true")},
        timeout=120,
    )


def _queue(workflow):
    data = json.dumps({"prompt": workflow, "client_id": str(uuid.uuid4())}).encode("utf-8")
    req = urllib.request.Request(f"http://{HOST}/prompt", data=data, headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
    if resp.get("node_errors"):
        raise RuntimeError("workflow validation failed: " + json.dumps(resp["node_errors"])[:1000])
    return resp["prompt_id"]


def _wait(prompt_id):
    for _ in range(config.POLL_MAX_RETRIES):
        history = requests.get(f"http://{HOST}/history/{prompt_id}", timeout=15).json()
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            for message in status.get("messages", []):
                if message[0] == "execution_error":
                    raise RuntimeError("ComfyUI execution error: " + json.dumps(message[1])[:1000])
            if entry.get("outputs"):
                return entry["outputs"]
        time.sleep(config.POLL_INTERVAL_MS / 1000)
    raise TimeoutError("render timed out")


def _output_path(outputs):
    for node_output in outputs.values():
        for video in node_output.get("gifs", []):
            return os.path.join(config.COMFY_OUTPUT_PATH, video.get("subfolder", ""), video["filename"])
    raise RuntimeError("no video output produced")


def _hardcut(paths, out_path):
    list_path = out_path + ".txt"
    with open(list_path, "w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"file '{path}'\n")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", out_path],
            check=True, capture_output=True,
        )
    return out_path


def _stitch(clips, durations, out_path, transition, trans_dur, has_audio):
    if len(clips) == 1:
        import shutil
        shutil.copy(clips[0], out_path)
        return out_path
    if transition == "cut":
        return _hardcut(clips, out_path)

    overlap = min(trans_dur, max(0.1, min(durations) - 0.1))
    inputs = []
    for clip in clips:
        inputs += ["-i", clip]

    video_chain, prev, cursor = [], "[0:v]", durations[0]
    for i in range(1, len(clips)):
        offset = max(0.0, cursor - overlap)
        label = f"[v{i}]"
        video_chain.append(f"{prev}[{i}:v]xfade=transition={transition}:duration={overlap:.3f}:offset={offset:.3f}{label}")
        prev, cursor = label, cursor + durations[i] - overlap

    filter_complex = ";".join(video_chain)
    maps = ["-map", prev]
    if has_audio:
        audio_chain, aprev = [], "[0:a]"
        for i in range(1, len(clips)):
            alabel = f"[a{i}]"
            audio_chain.append(f"{aprev}[{i}:a]acrossfade=d={overlap:.3f}{alabel}")
            aprev = alabel
        filter_complex += ";" + ";".join(audio_chain)
        maps += ["-map", aprev]

    cmd = ["ffmpeg", "-y"] + inputs + ["-filter_complex", filter_complex] + maps + \
        ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", out_path]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        return _hardcut(clips, out_path)
    return out_path


def _safe(text):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(text))


def handler(job):
    job_input = job.get("input") or {}
    ideas = job_input.get("ideas")
    if not ideas and job_input.get("scenes"):
        ideas = [{"id": "idea-0", "scenes": job_input["scenes"]}]
    if not ideas:
        return {"error": "missing 'ideas' (or 'scenes')"}

    defaults = job_input.get("defaults", {})
    _wait_for_models()
    base = _load_base()
    _upload_image(config.BLANK_IMAGE_NAME, _png(512, 512))

    outputs = []
    for idea_index, idea in enumerate(ideas):
        idea_id = _safe(idea.get("id", f"idea-{idea_index}"))
        scenes = idea.get("scenes", [])

        total_seconds = sum(resolve_seconds(scene, defaults)[2] for scene in scenes)
        if total_seconds > config.MAX_TOTAL_SECONDS:
            outputs.append({
                "idea_id": idea_id, "status": "error",
                "error": f"total duration {round(total_seconds, 2)}s exceeds max {config.MAX_TOTAL_SECONDS}s",
            })
            continue

        clips, durations, audio_flags, metas = [], [], [], []
        try:
            for scene_index, scene in enumerate(scenes):
                image_name = None
                keyframe = scene.get("keyframe") or scene.get("image")
                if keyframe:
                    if isinstance(keyframe, str) and keyframe.startswith("data:"):
                        keyframe = keyframe.split(",", 1)[1]
                    image_name = f"kf_{idea_id}_{scene_index}.png"
                    _upload_image(image_name, base64.b64decode(keyframe))

                prefix = f"ltx_{idea_id}_{scene_index}"
                workflow, meta = build_scene_workflow(base, scene, defaults, image_name=image_name, filename_prefix=prefix)
                meta["scene_index"] = scene_index
                clips.append(_output_path(_wait(_queue(workflow))))
                durations.append(meta["seconds"])
                audio_flags.append(bool(scene.get("audio", defaults.get("audio", False))))
                metas.append(meta)
        except Exception as error:
            outputs.append({"idea_id": idea_id, "status": "error", "error": str(error), "scenes_done": len(clips)})
            continue

        if config.STITCH_PER_IDEA and len(clips) > 1:
            transition = config.resolve_transition(idea.get("transition") or defaults.get("transition"))
            trans_dur = float(idea.get("transition_duration") or defaults.get("transition_duration") or config.TRANSITION_DURATION)
            movie = os.path.join(config.COMFY_OUTPUT_PATH, f"movie_{idea_id}.mp4")
            results = [_stitch(clips, durations, movie, transition, trans_dur, all(audio_flags))]
        else:
            results = clips

        entry = {"idea_id": idea_id, "status": "success", "scenes": len(clips), "meta": metas}
        if config.RETURN_MODE == "base64":
            entry["videos"] = [
                {"filename": os.path.basename(p), "data": base64.b64encode(open(p, "rb").read()).decode("utf-8")}
                for p in results
            ]
            entry["mime"] = "video/mp4"
        else:
            entry["paths"] = results
        outputs.append(entry)

    return {"outputs": outputs}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
