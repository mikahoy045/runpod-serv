# LTX-2.3 Batch Movie Worker (Runpod Serverless)

A Runpod Serverless worker that turns your scene generator's JSON into **short movies**, on a single 48GB GPU.

You send a **batch of ideas**; each idea is a list of **scenes** (`prompt` + `camera_motion` + optional `keyframe`). The worker generates one clip per scene and **concatenates the scenes of each idea into one mp4 movie** — loading the model **once per request** so a batch of 12–20 scenes amortizes the cold start.

Built on ComfyUI + **LTX-2.3 Distilled v1.1** (fp8), the camera-control LoRAs, two-stage upscaling, and `LTXVChunkFeedForward`. The v1.1 fp8 transformer is ~24GB, so a **48GB GPU (A40)** is required — 24GB cards (4090/3090/A5000) OOM.

## What it does per scene

- **Text-to-video or image-to-video** — if the scene has a `keyframe`, it's I2V (the keyframe is the start frame); if not, it's T2V. (Handled by the `LTXVImgToVideoInplace` `bypass` flag.)
- **Camera control** — `camera_motion` selects an LTX-2 camera LoRA: `dolly_in`, `dolly_out`, `dolly_left`, `dolly_right`, `jib_up`, `jib_down`, `static`. The motion is also woven into the text prompt.
- **Two-stage upscale** — base render → 2× spatial upscaler (v1.1) → tiled VAE decode.
- **Optional audio** — LTX-2.3 synchronized audio, off by default.

Then the scenes of an idea are concatenated → one movie mp4 per idea.

## Validated on a live GPU

This was validated end-to-end on the live **A40 (48GB)** Runpod Serverless endpoint before shipping:
- ✅ I2V (keyframe) render → mp4
- ✅ T2V (no keyframe, `bypass`) + camera LoRA → mp4
- ✅ `LTXVChunkFeedForward` graft (VRAM fit) → mp4
- ✅ `scene_workflow.py` patching accepted with zero node errors
- ✅ 2-scene idea → 2 clips → **concatenated movie** mp4

The graph is auto-reconciled against the live `/object_info` at worker startup, so it tolerates ComfyUI-LTXVideo node-version drift.

## Input schema

```jsonc
{
  "input": {
    "defaults": {                      // applied to every scene unless overridden
      "aspect_ratio": "9:16",          // or "16:9"
      "resolution": "720p",            // 480p | 576p | 720p | 1080p | 1440p (base; final is 2x)
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

## How the worker boots (lean image + network volume)

The image is **lean** — ComfyUI + the four custom nodes (`ComfyUI-LTXVideo`, `VideoHelperSuite`, `KJNodes`, `rgthree-comfy`) + the batch handler, built on `nvidia/cuda:12.8.1-cudnn` — with **no models baked in** (~12GB, vs ~60GB if baked). The full model set (~56GB: the v1.1 fp8 transformer, the Gemma-3 text encoder, the video/audio VAEs, the v1.1 spatial upscaler, the taehv preview VAE, and the 7 camera LoRAs) is **downloaded once at first boot** from Hugging Face with `aria2c -x16 -s16` (wget fallback) onto an attached **network volume** at `/runpod-volume/models`, where it persists across every worker.

To avoid Runpod's ~9-minute init-kill loop, [`worker/start.sh`](worker/start.sh) registers the handler **immediately** and downloads models in the **background** (`populate_models &`); each job then waits on a `${MODELS_ROOT}/.models_ready` marker (`_wait_for_models`, timeout `LTX_MODELS_READY_TIMEOUT`). So the **first** request against a fresh volume blocks ~5 min while models land; every request after that is fast (models already on the volume, FlashBoot warm).

> **48GB GPU required.** The fp8 transformer alone is ~24GB, so it will not fit a 24GB card. **A40 (48GB)** is the reliable pick, and — for network-volume serverless — is currently only offered in the **EU-SE-1** data center, so the volume *and* the endpoint must both live there.

## Deploy A — Runpod Console (GitHub build)

The image builds server-side from the [`Dockerfile`](Dockerfile) — no local Docker needed. A human, a Console/computer-use agent, or (for steps 1 + 3–5) the Runpod MCP can follow this verbatim.

**Preflight:** the account needs **billing enabled** (workers won't start otherwise), **GitHub connected**, and **A40 in stock in EU-SE-1** — the only data center that offers A40 together with network volumes.

1. **Create the network volume FIRST.** Runpod → Storage → **Network Volumes** → New → **EU-SE-1**, **≥70GB**. *Order matters:* attaching a volume pins the endpoint to that volume's data center, and only GPUs in that DC become selectable — creating the EU-SE-1 volume first is what makes **A40** available when you build the endpoint.
2. **Build the endpoint from the repo.** Serverless → **New Endpoint → Import Git Repository** → select the repo, branch `main`, Dockerfile path `Dockerfile`. Runpod builds the lean image and pushes it to `registry.runpod.net`. *(This build step is the one thing the MCP can't do — it needs a repo/Console build or a pre-pushed public image; see Deploy B.)*
3. **Configure the machine.** Attach the volume from step 1, pick **A40 48GB**, **Workers Min 0 / Max N**, **idle 5s**, **exec timeout 3600s**, **container disk ~20–30GB** (models live on the volume, not the container). Env vars optional — defaults are sane (see table below).
4. **Deploy, then warm it.** Send one job; the **first** request downloads ~56GB of models to the volume (~5 min). Watch the worker logs for `ltx-batch: ALL MODELS READY`. Every request after that is fast.
5. **Use it.** `Send a batch` (below) with the new endpoint id + your API key.

## Deploy B — Provision via Runpod MCP (agent-driven)

An AI agent with the **Runpod MCP** can stand the whole endpoint up. The one step MCP can't do is trigger a GitHub build, so first make the image **pullable**: either reuse the `registry.runpod.net/...` ref Runpod produced from a Console GitHub build (Deploy A step 3), or `docker build` this repo and push it to a registry (GHCR / Docker Hub).

Then the agent runs these MCP calls in order:

1. **Find the GPU pool + region.** `list-gpu-types` with `minMemoryGb: 48` → note the A40 pool name (`AMPERE_48`). `list-data-centers` to confirm A40 + network volumes are offered in **EU-SE-1**. Do **not** pick a 24GB pool.
2. **Create the volume.** `create-network-volume` → `{ name: "ltx-models", size: 70, dataCenterId: "EU-SE-1" }`. Keep the returned volume id.
3. **Create the endpoint (v2).** `create-endpoint`:
   ```jsonc
   {
     "name": "ltx-a40",
     "imageName": "<your pullable image ref>",
     "gpuPoolIds": ["AMPERE_48"],           // pool name from list-gpu-types, NOT a gpuTypeId
     "networkVolumeIds": ["<volume id from step 2>"],
     "dataCenterIds": ["EU-SE-1"],
     "workersMin": 0,
     "workersMax": 3,
     "idleTimeout": 5,
     "executionTimeoutMs": 3600000,
     "flashboot": "FLASHBOOT",
     "containerDiskInGb": 30,
     "env": {
       "LTX_DEFAULT_ASPECT_RATIO": "9:16",
       "LTX_DEFAULT_RESOLUTION_TIER": "720p"
       // ...any other LTX_* overrides from the env table below
     }
   }
   ```
   For a **private** image, first `create-container-registry-auth` and pass its id as `containerRegistryAuthId`. The network volume auto-mounts at `/runpod-volume` — no mount path needed.
4. **Warm it.** Submit one small job (`run-endpoint` / the client below). The worker downloads models to the volume in the background (~5 min) and the job completes once `.models_ready` flips; watch with `get-endpoint` / `endpoint-health` and the worker logs (`ltx-batch: ALL MODELS READY`). After that the endpoint is ready for real batches.

> Gotchas: the volume, GPU pool, and endpoint **must share the EU-SE-1 data center**. Serverless telemetry may show *"Volume usage: No volume"* even when it's mounted — trust the `ltx-batch: done ...` container logs, not the widget. Runpod MCP `delete-network-volume` / `delete-endpoint` return *"Unexpected end of JSON input"* on **success** (204, empty body) — verify with the matching `list-*` call rather than treating it as an error.

## Send a batch

```bash
export RUNPOD_API_KEY=...           # your Runpod key
export RUNPOD_ENDPOINT_ID=...       # the deployed endpoint id
python client/generate.py --input samples/egypt_1500bc.json --out-dir output
```

The client base64-encodes any `keyframe` file paths, submits the batch, polls, and saves each idea's movie to `output/`. Sample batches live in [`samples/`](samples) (`egypt_1500bc`, `egypt_2026`, `newyork_2026`).

## Cost

- **Serverless** bills per second of actual generation only (idle = $0). A40 48GB serverless ≈ $0.00034/s.
- **First boot on a fresh volume** downloads ~56GB of models (~5 min via `aria2c`) — a one-time cost; the volume persists them for every later request.
- A batch of N scenes pays **one** model load then renders each scene; the distilled v1.1 + 8-step sampler is fast.

## Environment variables (worker)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LTX_DEFAULT_ASPECT_RATIO` | `9:16` | Default aspect when a scene omits it |
| `LTX_DEFAULT_RESOLUTION_TIER` | `720p` | Default base resolution tier (`480p`–`1440p`) |
| `LTX_DEFAULT_FPS` | `24` | Default fps |
| `LTX_DEFAULT_DURATION_SECONDS` | `4` | Default per-scene length |
| `LTX_CAMERA_LORA_STRENGTH` | `0.8` | Camera LoRA strength |
| `LTX_CHUNK_FF_CHUNKS` | `2` | Chunk-feedforward chunks (0 disables; raise to fit smaller VRAM) |
| `LTX_STITCH_PER_IDEA` | `true` | Concatenate an idea's scenes into one movie |
| `LTX_RETURN_MODE` | `base64` | `base64` inline videos, or `volume` (paths only) |
| `LTX_TRANSITION` | `fade` | Default transition (`fade` / `dissolve` / `cut`; `crossfade` aliases to `fade`) |
| `LTX_TRANSITION_DURATION` | `0.5` | Crossfade overlap in seconds |
| `LTX_MAX_SCENE_SECONDS` | `10` | Per-scene duration cap |
| `LTX_MAX_TOTAL_SECONDS` | `30` | Per-idea total duration cap (over → error, no GPU spent) |
| `LTX_MAX_PROMPT_WORDS` | `200` | Prompt word cap |
| `LTX_VOLUME_ROOT` | `/runpod-volume` | Network-volume mount; models stage under `<root>/models` (falls back to `/comfyui/models` if absent) |
| `LTX_MODELS_READY_TIMEOUT` | `1800` | Seconds a job waits for the background model download before erroring |
| `LTX_HF_BASE` | `https://huggingface.co` | Hugging Face base URL for model downloads (set a mirror if needed) |

## Notes

- **v1.1** is used (`ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled` + `spatial-upscaler-x2-1.1`) — v1.0's limitations avoided.
- **Models are staged on the network volume, not baked into the image** (see *How the worker boots*). The full set is fetched from Hugging Face at first boot. To swap the transformer or any file, edit the `MODELS` list in [`worker/start.sh`](worker/start.sh) (and the matching `unet_name` in `worker/workflow_base.json`).
- **Camera LoRAs are LTX-2-19b** and partially compatible with 2.3 (strength ~0.6–0.8 works well).
- **Transitions are crossfade dissolves** (post-process blend between scene clips via ffmpeg `xfade`/`acrossfade`), default 0.5s. This is *not* true in-generation content morphing. For seamless prompt-morphing inside a single generation you'd need `PromptRelayEncodeTimeline` (not in the current node pack); for optical content morphs, frame interpolation (RIFE/FILM) — both are future upgrades.
- Credit: the ComfyUI base-stage build (comfy-cli, torch cu128, the custom-node set) follows [`e-dream-ai/gpu-container-ltx`](https://github.com/e-dream-ai/gpu-container-ltx) ("Jef's workflow"); this repo builds a lean image on top of `nvidia/cuda` and stages models on a volume instead of baking them in.
