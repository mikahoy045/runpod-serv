# LTX-2.3 Batch Movie Worker (Runpod Serverless)

A Runpod Serverless worker that turns your scene generator's JSON into **short movies**, on a budget GPU.

You send a **batch of ideas**; each idea is a list of **scenes** (`prompt` + `camera_motion` + optional `keyframe`). The worker generates one clip per scene and **concatenates the scenes of each idea into one mp4 movie** — loading the model **once per request** so a batch of 12–20 scenes amortizes the cold start.

Built on ComfyUI + **LTX-2.3 Distilled v1.1** (fp8), the camera-control LoRAs, two-stage upscaling, and `LTXVChunkFeedForward` so the 22B model fits a 24GB **RTX 4090**.

## What it does per scene

- **Text-to-video or image-to-video** — if the scene has a `keyframe`, it's I2V (the keyframe is the start frame); if not, it's T2V. (Handled by the `LTXVImgToVideoInplace` `bypass` flag.)
- **Camera control** — `camera_motion` selects an LTX-2 camera LoRA: `dolly_in`, `dolly_out`, `dolly_left`, `dolly_right`, `jib_up`, `jib_down`, `static`. The motion is also woven into the text prompt.
- **Two-stage upscale** — base render → 2× spatial upscaler (v1.1) → tiled VAE decode.
- **Optional audio** — LTX-2.3 synchronized audio, off by default.

Then the scenes of an idea are concatenated → one movie mp4 per idea.

## Validated on a live GPU

This was validated end-to-end on a Runpod Pod (RTX-class GPU) before shipping:
- ✅ I2V (keyframe) render → mp4
- ✅ T2V (no keyframe, `bypass`) + camera LoRA → mp4
- ✅ `LTXVChunkFeedForward` graft (the 4090 fit) → mp4
- ✅ `scene_workflow.py` patching accepted with zero node errors
- ✅ 2-scene idea → 2 clips → **concatenated movie** mp4

The graph is auto-reconciled against the live `/object_info` at worker startup, so it tolerates ComfyUI-LTXVideo node-version drift.

## Input schema

```jsonc
{
  "input": {
    "defaults": {                      // applied to every scene unless overridden
      "aspect_ratio": "9:16",          // or "16:9"
      "resolution": "720p",            // 480p | 576p | 720p | 1080p (base; final is 2x)
      "fps": 24,
      "style": "real",                 // real | cinematic | anime | 3dpixar | claymation | ... or free text
      "transition": "crossfade",       // crossfade (default) | dissolve | cut
      "transition_duration": 0.5,      // seconds of crossfade overlap
      "audio": false,
      "negative_prompt": "..."
    },
    "ideas": [
      {
        "id": "egypt-1500bc",
        "scenes": [
          { "prompt": "...", "camera_motion": "dolly_in" },                 // T2V
          { "prompt": "...", "camera_motion": "jib_up",
            "keyframe": "data:image/png;base64,..." },                       // I2V
          { "prompt": "...", "camera_motion": "static", "duration_seconds": 6 }
        ]
      }
      // ... 3–5 ideas in one request → one model load for the whole batch
    ]
  }
}
```

Per-scene overrides: `prompt` (required), `camera_motion`, `keyframe` (base64 / data-uri), `negative_prompt`, `style`, `aspect_ratio`, `resolution`, `width`/`height`, `duration_seconds`/`num_frames`, `fps`, `seed`, `audio`, `camera_strength`. Resolution = base (stage-1) dims (multiple of 32); final video is 2×. Frames snap to `8k+1`.

**Flexible structure:**
- **Any number of scenes** per idea — 3, 4, 6, whatever your generator produces.
- **Variable per-scene length** — set `duration_seconds` per scene. Each scene is capped at `LTX_MAX_SCENE_SECONDS` (10s) and the **total per idea must be ≤ `LTX_MAX_TOTAL_SECONDS` (30s)** — an idea over the total is returned as an error (no GPU wasted).
- **Style presets** (`style`): `real`, `cinematic`, `anime`, `3dpixar`/`pixar`, `3d`, `claymation`, `comic`, `watercolor`, `cyberpunk`, `vintage` — or pass free text. Set once in `defaults` or per scene. (LTX is realism-first; heavy stylization like anime is best-effort via prompt — a style LoRA would strengthen it.)
- **Transitions** (`transition`): `crossfade` (default dissolve), `dissolve`, or `cut` (hard cut). `transition_duration` controls the overlap. Set on `defaults` or per idea.
- **Keyframe per scene** — give a scene a `keyframe` (the start frame) for image-to-video; omit it for text-to-video. Mix freely within one idea.

**Output:**
```json
{ "outputs": [ { "idea_id": "egypt-1500bc", "status": "success", "scenes": 4,
                 "videos": [ { "filename": "movie_egypt-1500bc.mp4", "data": "<base64>" } ],
                 "meta": [ ... per-scene ... ] } ] }
```
Keep scenes within an idea at the **same resolution/aspect** so the concat is a clean stream copy.

## Deploy (GitHub → Runpod, no local Docker needed)

The worker image is built from the [`Dockerfile`](Dockerfile) — it extends the public `ghcr.io/e-dream-ai/gpu-container-ltx` image (ComfyUI + LTX-2.3 + camera LoRAs + VAEs + v1.1 upscaler) and adds the **v1.1 distilled transformer** + the batch handler.

1. Push this repo to GitHub.
2. Runpod console → **Serverless → New Endpoint → Import Git Repository** → select the repo (Dockerfile path `Dockerfile`). Runpod builds the image server-side.
3. **GPU:** RTX 4090 (24GB) is enough thanks to fp8 + chunk-feedforward + tiled decode. L40S/A100 for more headroom/speed.
4. **Workers:** Min 0 (scale to zero) or Min 1 during a busy session to skip cold starts. Container disk ~60GB (the v1.1 transformer is baked in).
5. Send a batch:

```bash
export RUNPOD_API_KEY=...           # your Runpod key
export RUNPOD_ENDPOINT_ID=...       # the deployed endpoint id
python client/generate.py --input samples/egypt_1500bc.json --out-dir output
```

The client base64-encodes any `keyframe` file paths, submits the batch, polls, and saves each idea's movie to `output/`.

## Cost

- **Serverless** bills per second of actual generation only (idle = $0). 4090 serverless ≈ $0.00031/s.
- A batch of N scenes pays **one** model load (~seconds–minute) then renders each scene; the distilled v1.1 + 8-step sampler is fast.

## Environment variables (worker)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LTX_DEFAULT_ASPECT_RATIO` | `9:16` | Default aspect when a scene omits it |
| `LTX_DEFAULT_RESOLUTION_TIER` | `720p` | Default base resolution tier |
| `LTX_DEFAULT_FPS` | `24` | Default fps |
| `LTX_DEFAULT_DURATION_SECONDS` | `4` | Default per-scene length |
| `LTX_CAMERA_LORA_STRENGTH` | `0.8` | Camera LoRA strength |
| `LTX_CHUNK_FF_CHUNKS` | `2` | Chunk-feedforward chunks (0 disables; raise to fit smaller VRAM) |
| `LTX_STITCH_PER_IDEA` | `true` | Concatenate an idea's scenes into one movie |
| `LTX_RETURN_MODE` | `base64` | `base64` inline videos, or `volume` (paths only) |

## Notes

- **v1.1** is used (`ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled` + `spatial-upscaler-x2-1.1`) — v1.0's limitations avoided. The model file is swappable via the `unet_name` in `worker/workflow_base.json`.
- **Camera LoRAs are LTX-2-19b** and partially compatible with 2.3 (strength ~0.6–0.8 works well); this matches how the base image ships them.
- **Transitions are crossfade dissolves** (post-process blend between scene clips via ffmpeg `xfade`/`acrossfade`), default 0.5s. This is *not* true in-generation content morphing. For seamless prompt-morphing inside a single generation you'd need `PromptRelayEncodeTimeline` (not in the current node pack); for optical content morphs, frame interpolation (RIFE/FILM) — both are future upgrades.
- Credit: the worker container is based on [`e-dream-ai/gpu-container-ltx`](https://github.com/e-dream-ai/gpu-container-ltx) ("Jef's workflow").
