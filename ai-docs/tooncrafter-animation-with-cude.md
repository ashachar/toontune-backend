# ToonCrafter — Complete Local Installation & Run Guide (Two‑Image Example)

> This guide shows you **exactly** how to install ToonCrafter locally and run it on **two images (start & end)** to produce a short animation. It includes environment checks, command‑line and Gradio usage, a clean file layout, and troubleshooting tips.

---

## What is ToonCrafter (one‑liner)
**ToonCrafter** generates the in‑between frames between two cartoon keyframes using diffusion‑based video priors — i.e., you give it a **start** and an **end** image, and it synthesizes a short video that transitions between them. citeturn1view0

---

## System Requirements & Limits (read first)
- **GPU:** NVIDIA CUDA GPU strongly recommended. The authors report ~**24 GB VRAM** (sometimes up to ~27 GB) in the reference implementation; the community has variants that run around **~10–12 GB** with pruned/fp16 workflows. citeturn1view0  
- **Frames & Resolution:** Official release supports up to **16 frames** at **512×320** (w×h) with default scripts. (You can change FPS and steps to trade speed/quality.) citeturn1view0
- **Model size:** The main checkpoint is large (~**10.5 GB**). Downloading it can take time and disk space. citeturn4view0

> **Tip:** If your GPU has <16 GB VRAM, consider trying the community fp16/pruned flows via ComfyUI as linked from the repo’s “Community support” section. citeturn1view0

---

## Quick Start (Linux/Windows using Conda)
The official repo provides a minimal setup with Anaconda + pip.

```bash
# 1) Clone the official repo
git clone https://github.com/Doubiiu/ToonCrafter.git
cd ToonCrafter

# 2) Create and activate a clean environment
# The README uses Python 3.8.5 — stick close to that unless you know why to change
conda create -n tooncrafter python=3.8.5 -y
conda activate tooncrafter

# 3) Install Python dependencies
pip install -r requirements.txt
```
citeturn1view0

### CUDA, PyTorch & ffmpeg
- Use a CUDA-enabled PyTorch build that matches your system drivers.  
- You’ll also want **ffmpeg** on PATH if you plan to encode frames into a video yourself.

**Pre‑flight checks:**
```bash
# See your GPU and VRAM
nvidia-smi

# Confirm PyTorch sees CUDA
python - << 'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (binary):", torch.version.cuda)
PY

# Check ffmpeg (optional but useful)
ffmpeg -version
```

---

## Get the Pretrained Model (mandatory)
Download the official **ToonCrafter_512** checkpoint and put it exactly where the scripts expect it:

```
ToonCrafter/
└── checkpoints/
    └── tooncrafter_512_interp_v1/
        └── model.ckpt  ← (≈10.5 GB)
```

- **Download source:** Hugging Face model file `model.ckpt`. citeturn4view0  
- **Expected path:** The provided `scripts/run.sh` looks for it at `checkpoints/tooncrafter_512_interp_v1/model.ckpt`. citeturn7view0

> **Heads‑up:** If the file/path differs, the script will fail with “checkpoint Not Found!”

---

## Prepare Your Two Images (clean, isolated prompt folder)
You will point ToonCrafter at a **prompt directory** containing:
1) your **start** image,  
2) your **end** image, and  
3) a minimal `prompts.txt` (one line per sample).

Create a fresh folder so you process **only** your images:

```
ToonCrafter/
└── prompts/
    └── my_two_frames_demo/
        ├── my_scene_frame1.png   # <-- START image (name must end with `_frame1`)
        ├── my_scene_frame3.png   # <-- END image   (name must end with `_frame3`)
        └── prompts.txt           # one text line matching your pair
```

- **Naming pattern matters:** the loader pairs files by the `_frame1` / `_frame3` suffix convention (see the project’s example prompt pack). citeturn16view0  
- **Example `prompts.txt`:**
  ```text
  walking man in an anime scene
  ```
  (The repo’s demo uses a simple `prompts.txt` with lines like “walking man” and “an anime scene”.) citeturn17view0

> **Tip:** Resize/crop both images to **512×320 (width×height)** for the default config, or pass `--width 512 --height 320` on the command line. citeturn1view0turn7view0

---

## Option A — Command‑line Inference (recommended for reproducibility)
You can call the underlying script directly (mirrors what `scripts/run.sh` does), while **overriding** the `--prompt_dir` to your custom folder.

```bash
# from the ToonCrafter repo root
CKPT=checkpoints/tooncrafter_512_interp_v1/model.ckpt
CONFIG=configs/inference_512_v1.0.yaml
OUTDIR=results
SEED=123

python3 scripts/evaluation/inference.py \
  --seed ${SEED} \
  --ckpt_path ${CKPT} \
  --config ${CONFIG} \
  --savedir ${OUTDIR}/tooncrafter_512_interp_seed${SEED} \
  --n_samples 1 \
  --bs 1 --height 320 --width 512 \
  --unconditional_guidance_scale 7.5 \
  --ddim_steps 50 \
  --ddim_eta 1.0 \
  --prompt_dir prompts/my_two_frames_demo \
  --text_input \
  --video_length 16 \
  --frame_stride 10 \
  --timestep_spacing 'uniform_trailing' \
  --guidance_rescale 0.7 \
  --perframe_ae \
  --interp
```

- The flags above follow the reference `run.sh` with `--prompt_dir` changed to your folder and `--frame_stride` set to **10** (default in `run.sh`). citeturn7view0  
- **Outputs:** Generated frames are saved under `results/.../samples_separate/`. (You can convert them to a video with `ffmpeg` if needed; see below.)

**Speed/quality knobs you can safely tweak:**
- `--ddim_steps` (e.g., 25 for faster runs). The authors note fewer steps reduce inference time. citeturn1view0
- `--frame_stride` (FS; lower = slower motion, higher = larger motion interval). The script comments: “smaller value → larger motion.” citeturn7view0
- `--video_length` (<= 16 by default release). citeturn1view0

**(Optional) Make an MP4 from frames:**
```bash
# Example: encode PNG frames in the output folder into a 5 FPS MP4
cd results/tooncrafter_512_interp_seed123/samples_separate

# If frames are named sequentially, something like:
ffmpeg -r 5 -pattern_type glob -i "*.png" -c:v libx264 -pix_fmt yuv420p output.mp4
```

---

## Option B — Local Gradio App (point‑and‑click)
A simple UI is included. First, ensure your checkpoint is in place (see above). Then:

```bash
# from the ToonCrafter repo root (env activated)
python gradio_app.py
```

- In your terminal, Gradio will print a local URL (e.g., `http://127.0.0.1:7860`).  
- Upload **Input Image 1** (start) and **Input Image 2** (end), type a prompt, set seed/ETA/FPS/etc., and click **Generate**. citeturn1view0turn11search1

---

## Windows‑specific note (optional helper)
There is a community Windows fork with **PowerShell installers** and a one‑click Gradio launcher. If you prefer that route, see:  
- `ToonCrafter-for-windows` (adds `install.ps1`, `run_gui.ps1`, prerequisites: Python 3.8–3.11, CUDA≥11.3, ffmpeg, git). citeturn5view0

> You still need the checkpoint; the fork documents that it will fetch/place models for you automatically in some flows. citeturn5view0

---

## Troubleshooting & Tips
- **OOM / VRAM errors:** Try reducing `--ddim_steps` (e.g., 25), shortening `--video_length` (e.g., 8), or experimenting with the community fp16/pruned workflows (ComfyUI) referenced in the repo’s “Community support”. citeturn1view0
- **Wrong resolution:** Keep **512×320** (w×h) unless you’ve modified configs accordingly. The default config & scripts assume that shape. citeturn1view0turn7view0
- **Checkpoint path errors:** Double‑check the exact path `checkpoints/tooncrafter_512_interp_v1/model.ckpt` and filename spelling. citeturn7view0
- **Prompt directory not found:** Verify the folder you pass via `--prompt_dir` exists and contains your two files `*_frame1.*` and `*_frame3.*` plus `prompts.txt`. See the **example prompt pack** in the repo’s Hugging Face Space. citeturn16view0turn17view0
- **Slow downloads:** The checkpoint is ~10.5 GB — use a stable connection or a download manager. citeturn4view0

---

## Why these defaults?
- The official `run.sh` calls `inference.py` with the exact flags mirrored above (seed, steps, scale, per‑frame AE, etc.). We changed only the `--prompt_dir` to keep your demo isolated. citeturn7view0
- The frame cap (≤16) and base resolution (512×320) are design limits of the public release at the time of writing. citeturn1view0

---

## References & Further Reading
- **GitHub — Doubiiu/ToonCrafter (official):** setup, models, CLI & Gradio entry points. citeturn1view0  
- **Hugging Face model (`model.ckpt` ~10.5 GB):** download location. citeturn4view0  
- **Sample prompts & file naming (Hugging Face Space folder):** see image pairs and `prompts.txt`. citeturn16view0turn17view0  
- **`scripts/run.sh` (all CLI flags used above):** reference run script. citeturn7view0
- **Paper / Project Page:** for method details and limitations (optional reading). citeturn0search1turn0search4

---

### Fast sanity test (no editing required)
If you just want to confirm everything works **before** using your own images, run the defaults:

```bash
# uses prompts/512_interp from the repo’s sample pack (if present in your checkout)
sh scripts/run.sh
```
This will render examples into `results/...`. You can then switch to your `prompts/my_two_frames_demo` as shown above. citeturn7view0

---

## Summary
You now have a complete, reproducible path to: (1) install ToonCrafter; (2) download and place the model; (3) prepare a **two‑image** prompt folder; and (4) generate frames/video via **CLI** or **Gradio**. If you hit VRAM limits, try fewer steps/frames or a community fp16 workflow. citeturn1view0
