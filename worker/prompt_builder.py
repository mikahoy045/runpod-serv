import config

CAMERA_MOTION_PHRASES = {
    "dolly_in": "slow dolly-in, the camera smoothly pushes toward the subject",
    "dolly_out": "slow dolly-out, the camera smoothly pulls back",
    "dolly_left": "the camera tracks smoothly to the left",
    "dolly_right": "the camera tracks smoothly to the right",
    "jib_up": "the camera rises smoothly in a vertical jib up",
    "jib_down": "the camera descends smoothly in a vertical jib down",
    "static": "static locked-off camera, no camera movement",
}


def _clean(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def _humanize(value):
    return _clean(value).replace("-", " ").replace("_", " ").strip()


def _as_sentence(text):
    text = _clean(text)
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def build_prompt(scene):
    scene_text = _clean(scene.get("prompt") or scene.get("scene") or scene.get("description"))
    if not scene_text:
        raise ValueError("each scene needs a 'prompt'")

    style = _clean(config.style_modifier(scene.get("style")))
    camera = scene.get("camera") if isinstance(scene.get("camera"), dict) else {}

    sentences = []

    framing = " ".join(p for p in [_humanize(camera.get("angle")), _humanize(camera.get("shot_type"))] if p).strip()
    if framing:
        sentences.append(_as_sentence(framing))

    motion = config.normalize_camera_motion(scene.get("camera_motion") or camera.get("movement"))
    if motion != "none":
        sentences.append(_as_sentence(CAMERA_MOTION_PHRASES.get(motion, _humanize(motion))))

    lens = _humanize(camera.get("lens") or camera.get("focal_length"))
    if lens:
        sentences.append(_as_sentence(f"shot on a {lens} lens"))

    sentences.append(_as_sentence(scene_text))
    if style:
        sentences.append(_as_sentence(style))

    prompt = " ".join(s for s in sentences if s).strip()
    words = prompt.split()
    if len(words) > config.MAX_PROMPT_WORDS:
        prompt = " ".join(words[: config.MAX_PROMPT_WORDS])
    return prompt
