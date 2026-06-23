import os


def _env(name, default=None):
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _env_int(name, default):
    try:
        return int(_env(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    try:
        return float(_env(name, default))
    except (TypeError, ValueError):
        return default


COMFY_HOST = _env("COMFY_HOST", "127.0.0.1:8188")
COMFY_OUTPUT_PATH = _env("COMFY_OUTPUT_PATH", "/comfyui/output")
COMFY_INPUT_PATH = _env("COMFY_INPUT_PATH", "/comfyui/input")

WORKFLOW_PATH = _env("LTX_WORKFLOW_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow_base.json"))

NODES = {
    "unet": "1",
    "upscaler": "5",
    "camera_lora": "6",
    "chunk_ff": "7",
    "positive": "10",
    "negative": "11",
    "conditioning": "12",
    "load_image": "20",
    "latent_video": "30",
    "latent_audio": "31",
    "img2vid_s1": "32",
    "img2vid_s2": "51",
    "scheduler": "40",
    "seed_s1": "42",
    "seed_s2": "62",
    "video_combine": "80",
    "audio_decode": "71",
}

SUPPORTED_CAMERA_MOTIONS = (
    "dolly_in", "dolly_out", "dolly_left", "dolly_right", "jib_up", "jib_down", "static",
)

CAMERA_ALIASES = {
    "push_in": "dolly_in", "zoom_in": "dolly_in", "push": "dolly_in",
    "pull_out": "dolly_out", "pull_back": "dolly_out", "zoom_out": "dolly_out",
    "truck_left": "dolly_left", "slide_left": "dolly_left", "pan_left": "dolly_left",
    "truck_right": "dolly_right", "slide_right": "dolly_right", "pan_right": "dolly_right",
    "crane_up": "jib_up", "boom_up": "jib_up", "tilt_up": "jib_up",
    "crane_down": "jib_down", "boom_down": "jib_down", "tilt_down": "jib_down",
    "locked": "static", "fixed": "static", "tripod": "static", "none": "none",
}

CAMERA_LORA_FILENAME = "ltx-2-19b-lora-camera-control-{}.safetensors"
CAMERA_LORA_STRENGTH = _env_float("LTX_CAMERA_LORA_STRENGTH", 0.8)

DIMENSION_MULTIPLE = 32
FRAME_MULTIPLE = 8

BASE_LONG_SIDE = {
    "480p": 512,
    "576p": 576,
    "720p": 640,
    "1080p": 960,
    "1440p": 1280,
}

DEFAULT_ASPECT_RATIO = _env("LTX_DEFAULT_ASPECT_RATIO", "9:16")
DEFAULT_RESOLUTION_TIER = _env("LTX_DEFAULT_RESOLUTION_TIER", "720p")
DEFAULT_FPS = _env_float("LTX_DEFAULT_FPS", 24.0)
DEFAULT_DURATION_SECONDS = _env_float("LTX_DEFAULT_DURATION_SECONDS", 4.0)
MAX_PROMPT_WORDS = _env_int("LTX_MAX_PROMPT_WORDS", 200)

CHUNK_FF_CHUNKS = _env_int("LTX_CHUNK_FF_CHUNKS", 2)
BLANK_IMAGE_NAME = "ltx_blank.png"

RETURN_MODE = _env("LTX_RETURN_MODE", "base64").lower()
STITCH_PER_IDEA = _env("LTX_STITCH_PER_IDEA", "true").lower() == "true"

TRANSITION = _env("LTX_TRANSITION", "fade").lower()
TRANSITION_DURATION = _env_float("LTX_TRANSITION_DURATION", 0.5)
MAX_TOTAL_SECONDS = _env_float("LTX_MAX_TOTAL_SECONDS", 30.0)
MAX_SCENE_SECONDS = _env_float("LTX_MAX_SCENE_SECONDS", 10.0)

TRANSITION_ALIASES = {
    "crossfade": "fade", "morph": "fade", "cross": "fade",
    "dissolve": "dissolve", "none": "cut", "hardcut": "cut", "cut": "cut",
}

STYLE_PRESETS = {
    "real": "photorealistic, cinematic, realistic natural lighting, film grain, shot on a cinema camera",
    "realistic": "photorealistic, cinematic, realistic natural lighting, film grain, shot on a cinema camera",
    "cinematic": "cinematic film look, dramatic lighting, shallow depth of field, anamorphic",
    "anime": "anime style, cel-shaded 2D animation, vibrant colors, clean line art, studio anime aesthetic",
    "3dpixar": "3D Pixar-style animation, stylized characters, soft global illumination, subsurface scattering, polished CGI render",
    "pixar": "3D Pixar-style animation, stylized characters, soft global illumination, subsurface scattering, polished CGI render",
    "3d": "stylized 3D render, CGI animation, soft studio lighting",
    "claymation": "claymation stop-motion style, soft clay textures, handcrafted look",
    "comic": "comic book style, bold ink outlines, halftone shading, flat colors",
    "watercolor": "watercolor painting style, soft color washes, paper texture",
    "cyberpunk": "cyberpunk aesthetic, neon lighting, futuristic, moody atmosphere",
    "vintage": "vintage film look, retro color grading, grainy, nostalgic",
}

POLL_INTERVAL_MS = _env_int("COMFY_POLLING_INTERVAL_MS", 500)
POLL_MAX_RETRIES = _env_int("COMFY_POLLING_MAX_RETRIES", 1200)


def normalize_camera_motion(value):
    if value is None:
        return "none"
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key in SUPPORTED_CAMERA_MOTIONS:
        return key
    return CAMERA_ALIASES.get(key, "none")


def camera_lora_filename(motion):
    return CAMERA_LORA_FILENAME.format(motion.replace("_", "-"))


def style_modifier(value):
    if not value:
        return ""
    text = str(value).strip()
    return STYLE_PRESETS.get(text.lower(), text)


def resolve_transition(value=None):
    text = str(value or TRANSITION).strip().lower()
    return TRANSITION_ALIASES.get(text, text)
