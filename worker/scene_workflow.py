import copy
import random

import config
from prompt_builder import build_prompt


def _snap(value, multiple):
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def snap_num_frames(value):
    k = max(1, int(round((float(value) - 1) / config.FRAME_MULTIPLE)))
    return k * config.FRAME_MULTIPLE + 1


def resolve_base_dims(aspect_ratio, tier, width=None, height=None):
    if width and height:
        return _snap(width, config.DIMENSION_MULTIPLE), _snap(height, config.DIMENSION_MULTIPLE)
    long_side = config.BASE_LONG_SIDE.get(str(tier).lower(), config.BASE_LONG_SIDE[config.DEFAULT_RESOLUTION_TIER])
    parts = str(aspect_ratio).replace("x", ":").replace("X", ":").split(":")
    w_ratio, h_ratio = float(parts[0]), float(parts[1])
    if h_ratio >= w_ratio:
        out_h, out_w = long_side, long_side * w_ratio / h_ratio
    else:
        out_w, out_h = long_side, long_side * h_ratio / w_ratio
    return _snap(out_w, config.DIMENSION_MULTIPLE), _snap(out_h, config.DIMENSION_MULTIPLE)


def _pick(scene, defaults, key, fallback):
    if scene.get(key) is not None:
        return scene[key]
    if defaults.get(key) is not None:
        return defaults[key]
    return fallback


def resolve_seconds(scene, defaults):
    fps = float(_pick(scene, defaults, "fps", config.DEFAULT_FPS))
    if scene.get("num_frames"):
        frames = snap_num_frames(scene["num_frames"])
    else:
        frames = snap_num_frames(float(_pick(scene, defaults, "duration_seconds", config.DEFAULT_DURATION_SECONDS)) * fps)
    cap = snap_num_frames(config.MAX_SCENE_SECONDS * fps)
    frames = min(frames, cap)
    return frames, fps, round((frames - 1) / fps, 3)


def build_scene_workflow(base, scene, defaults, image_name=None, filename_prefix="ltx_scene"):
    wf = copy.deepcopy(base)
    n = config.NODES

    wf[n["positive"]]["inputs"]["text"] = build_prompt({**defaults, **scene})
    wf[n["negative"]]["inputs"]["text"] = (
        scene.get("negative_prompt")
        or defaults.get("negative_prompt")
        or "worst quality, blurry, distorted, watermark, text, low quality, deformed"
    )

    motion = config.normalize_camera_motion(scene.get("camera_motion") or (scene.get("camera") or {}).get("movement"))
    strength = float(_pick(scene, defaults, "camera_strength", config.CAMERA_LORA_STRENGTH))
    wf[n["camera_lora"]]["inputs"]["lora_01"] = {
        "on": motion != "none",
        "lora": config.camera_lora_filename(motion if motion != "none" else "static"),
        "strength": strength,
    }

    aspect = _pick(scene, defaults, "aspect_ratio", config.DEFAULT_ASPECT_RATIO)
    tier = _pick(scene, defaults, "resolution", config.DEFAULT_RESOLUTION_TIER)
    width, height = resolve_base_dims(aspect, tier, scene.get("width"), scene.get("height"))
    frames, fps, _ = resolve_seconds(scene, defaults)

    wf[n["latent_video"]]["inputs"].update({"width": width, "height": height, "length": frames})
    wf[n["latent_audio"]]["inputs"].update({"frames_number": frames, "frame_rate": fps})
    wf[n["conditioning"]]["inputs"]["frame_rate"] = fps
    wf[n["video_combine"]]["inputs"]["frame_rate"] = fps
    wf[n["video_combine"]]["inputs"]["filename_prefix"] = filename_prefix

    seed = int(_pick(scene, defaults, "seed", 0))
    if seed <= 0:
        seed = random.randint(1, 2**31 - 1)
    wf[n["seed_s1"]]["inputs"]["noise_seed"] = seed
    wf[n["seed_s2"]]["inputs"]["noise_seed"] = seed

    has_image = bool(image_name)
    wf[n["img2vid_s1"]]["inputs"]["bypass"] = not has_image
    wf[n["img2vid_s2"]]["inputs"]["bypass"] = not has_image
    wf[n["load_image"]]["inputs"]["image"] = image_name or config.BLANK_IMAGE_NAME

    if not bool(_pick(scene, defaults, "audio", False)):
        wf[n["video_combine"]]["inputs"].pop("audio", None)

    if config.CHUNK_FF_CHUNKS and n["chunk_ff"] in wf:
        wf[n["chunk_ff"]]["inputs"]["chunks"] = config.CHUNK_FF_CHUNKS
    elif n["chunk_ff"] in wf:
        wf[n["camera_lora"]]["inputs"]["model"] = [n["unet"], 0]
        wf.pop(n["chunk_ff"], None)

    meta = {
        "base_width": width,
        "base_height": height,
        "final_width": width * 2,
        "final_height": height * 2,
        "num_frames": frames,
        "fps": fps,
        "seconds": round((frames - 1) / fps, 3),
        "seed": seed,
        "camera_motion": motion,
        "mode": "i2v" if has_image else "t2v",
    }
    return wf, meta
