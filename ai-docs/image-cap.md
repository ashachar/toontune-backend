# AutoCap on macOS — Install & Use Guide (Apple Silicon & Intel)

> **Goal:** get the open-source **AutoCap** tool running on your Mac and generate high-quality captions for image datasets (great for LoRA training). This guide is beginner-friendly and fully reproducible.

---

## 1) What you'll install

- **AutoCap** (`yuval-avidani/autocap`) — Python CLI that captions images with Microsoft's **Florence-2-large** vision-language model.
- **Python 3.10+** (works with 3.11/3.12 too)
- Optional: **GPU acceleration** on Apple Silicon via **MPS** (Metal Performance Shaders). Falls back to CPU if unavailable.

> **Download size note:** the first run downloads the Florence-2-large model (~1.5 GB). Make sure you have a stable connection and free disk space.

---

## 2) System requirements

- macOS 13+ recommended (macOS 12 works for CPU; MPS works best on 13+)
- 8 GB RAM minimum (16 GB+ recommended)
- Python 3.10 or newer
- Apple Silicon (M1/M2/M3) or Intel Mac
- Git and Homebrew (optional, but this guide uses Homebrew for convenience)

---

## 3) Install prerequisites (one-time)

Open **Terminal** and run the following exactly as written.

### 3.1 Install Xcode Command Line Tools (compilers, headers)
```bash
xcode-select --install || true
```

### 3.2 Install Homebrew (skip if you already have it)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 3.3 Ensure Git and Python 3
```bash
brew install git python@3.11
```

Check versions:
```bash
git --version
python3 --version
```

---

## 4) Create a clean Python environment

Using a project folder keeps things tidy and avoids conflicts.

```bash
# Create and enter a workspace folder (any path is fine)
mkdir -p ~/autocap-work && cd ~/autocap-work

# Create an isolated virtual environment
python3 -m venv .venv

# Activate it for this Terminal session
source .venv/bin/activate

# Upgrade core packaging tools
python -m pip install --upgrade pip setuptools wheel
```

Every time you open a new Terminal window to use AutoCap, re-activate with:
```bash
cd ~/autocap-work
source .venv/bin/activate
```

---

## 5) Get AutoCap and install dependencies

```bash
# Clone the repository
git clone https://github.com/yuval-avidani/autocap.git
cd autocap

# Install Python dependencies
pip install -r requirements.txt
```

If PyTorch didn't install for some reason, install it explicitly:
```bash
pip install torch torchvision torchaudio
```

---

## 6) (Optional) Verify Apple Silicon GPU (MPS) support

If you have an M-series Mac and macOS 13+, you can accelerate inference with MPS.

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
PY
```

- If **MPS available: True** → you're set. AutoCap's `--device auto` will pick it up.
- If **False**, don't worry — AutoCap will use CPU.

---

## 7) Quick start: caption a folder

Place a few test images in a folder (e.g., `~/Pictures/autocap-demo`) and run:

```bash
# From the repo directory: ~/autocap-work/autocap
python autocap.py ~/Pictures/autocap-demo
```

**What happens:** for each image `image_001.jpg`, AutoCap writes `image_001.txt` with the generated caption in the same folder (unless you set `--output`). The first image may take longer due to model download & initialization.

---

## 8) Recommended usage patterns

### 8.1 General, detailed captions (default)
```bash
python autocap.py ~/Pictures/autocap-demo \
  --mode general \
  --task DETAILED_CAPTION
```

### 8.2 Maximum detail (heavier, best for rich datasets)
```bash
python autocap.py ~/Pictures/autocap-demo \
  --mode detailed \
  --task MORE_DETAILED_CAPTION
```

### 8.3 Style LoRA (focus on artistic style; removes specific objects)
```bash
python autocap.py ~/Pictures/art-style \
  --mode style \
  --trigger mystyle \
  --prepend-trigger \
  --task DETAILED_CAPTION
```

### 8.4 Character LoRA (pose, expression, stable descriptors)
```bash
python autocap.py ~/Pictures/characters \
  --mode character \
  --trigger charactername \
  --task MORE_DETAILED_CAPTION
```

### 8.5 Object LoRA (clean product/object descriptions)
```bash
python autocap.py ~/Pictures/objects \
  --mode object \
  --trigger myobject \
  --task MORE_DETAILED_CAPTION
```

### 8.6 Write captions to a separate output folder
```bash
python autocap.py ~/Pictures/autocap-demo \
  --output ~/Pictures/autocap-captions
```

---

## 9) Command reference (most useful flags)

| Flag | Meaning | Default |
|------|---------|---------|
| `input_dir` | Folder with images | required |
| `--output`, `-o` | Output folder for .txt captions | same as input |
| `--mode`, `-m` | general, style, character, object, detailed, simple | general |
| `--task`, `-t` | CAPTION, DETAILED_CAPTION, MORE_DETAILED_CAPTION | DETAILED_CAPTION |
| `--trigger` | Trigger word added to all captions | none |
| `--prepend-trigger` | Put trigger at start | True |
| `--append-trigger` | Put trigger at end | False |
| `--remove-objects` | Comma-separated words to drop from captions | [] |
| `--max-length` | Max caption length (characters) | 300 |
| `--device` | auto, cuda, cpu, mps | auto |
| `--no-fp16` | Disable float16 (use float32) | False |
| `--overwrite` | Overwrite existing .txt files | False |
| `--no-skip` | Process even if caption exists | False |
| `--config`, `-c` | Load settings from JSON | none |
| `--save-config` | Save current flags to JSON | none |
| `--verbose`, `-v` | Verbose logging | False |

**Example:** remove watermarks & site names from captions
```bash
python autocap.py ~/Pictures/shop \
  --mode object \
  --task DETAILED_CAPTION \
  --remove-objects "watermark,logo,url,website"
```

---

## 10) Reusable configuration (JSON)

Save your favorite settings once, reuse forever:

```bash
python autocap.py ~/Pictures/objects \
  --mode object \
  --task MORE_DETAILED_CAPTION \
  --trigger myobject \
  --prepend-trigger \
  --save-config ~/autocap-work/my_object_lora.json
```

Run later with:
```bash
python autocap.py ~/Pictures/objects --config ~/autocap-work/my_object_lora.json
```

**Sample `my_object_lora.json`:**
```json
{
  "mode": "object",
  "task": "MORE_DETAILED_CAPTION",
  "trigger_word": "myobject",
  "prepend_trigger": true,
  "remove_objects": [],
  "max_length": 300,
  "device": "auto",
  "fp16": false,
  "skip_existing": true
}
```

---

## 11) Performance tips (Mac specific)

- **Apple Silicon (M-series):** prefer `--device auto` (will pick mps) or set `--device mps` explicitly.
- **Speed vs detail:** `MORE_DETAILED_CAPTION` is slower than `DETAILED_CAPTION`. For large datasets, start with `DETAILED_CAPTION`.
- **Memory stability:** if you hit out-of-memory, try `--no-fp16` off first (FP16 saves memory). If instability occurs, enable `--no-fp16` (uses FP32, slower but stable).
- **Batching strategy:** AutoCap processes efficiently and skips already-captioned images unless `--no-skip` is set.
- **First-run slowness:** model download + first load is always the slowest.

---

## 12) Logging & stats

- Real-time progress with ETA
- Summary stats at the end
- Add `--verbose` to print more details
- A log file (`autocap.log`) may be created for deeper debugging

---

## 13) Troubleshooting

### A) Install errors related to transformers / deps
```bash
pip install "transformers>=4.45.0" einops timm
```

### B) PyTorch install issues on macOS
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
```

### C) Extremely slow performance
- Confirm you're on M-series + macOS 13+ and that MPS is available (Section 6).
- Use `--task DETAILED_CAPTION` instead of `MORE_DETAILED_CAPTION` for faster runs.
- Close heavy apps (Chrome tabs, video editors, etc.) to free RAM.

### D) "File not found" or no captions appear
- Ensure your `input_dir` is correct and contains supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`, `.tiff`.

### E) Force CPU for reproducibility
```bash
python autocap.py ~/Pictures/autocap-demo --device cpu
```

---

## 14) Keeping AutoCap updated

```bash
# From the repo dir
git pull
pip install -r requirements.txt --upgrade
```

---

## 15) Uninstall / clean up

```bash
# Remove the project checkout
rm -rf ~/autocap-work/autocap

# (Optional) delete the virtual environment
rm -rf ~/autocap-work/.venv
```

Model files are cached by Hugging Face under `~/.cache` (or similar). You can remove them if you need disk space, but they will re-download next run.

---

## 16) One-line smoke test (end-to-end)

```bash
# Activate env and run a dry caption on a folder (replace with a folder that contains at least 1 JPG/PNG)
cd ~/autocap-work && source .venv/bin/activate && cd autocap && \
python autocap.py ~/Pictures/autocap-demo --mode general --task DETAILED_CAPTION --device auto --verbose
```

You should see progress logs and `.txt` files created next to your images (or in `--output` if specified).

---

## 17) FAQ

**Q: Do I need an NVIDIA GPU?**  
A: No. On Macs, Apple Silicon uses MPS. Intel Macs run on CPU.

**Q: Can I resume a large dataset?**  
A: Yes. AutoCap skips images that already have caption files unless you set `--no-skip` or `--overwrite`.

**Q: Where do captions go?**  
A: By default, next to each image as `image_name.txt`. Or set `--output`.

**Q: Does it work offline?**  
A: After the model has downloaded once, you can caption offline. First run requires internet.

---

## 18) Minimal quick-use cheat-sheet

```bash
# First time only
xcode-select --install || true
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install git python@3.11

# Project setup
mkdir -p ~/autocap-work && cd ~/autocap-work
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
git clone https://github.com/yuval-avidani/autocap.git
cd autocap
pip install -r requirements.txt

# Run
python autocap.py ~/Pictures/autocap-demo --mode general --task DETAILED_CAPTION --device auto
```

**You're all set!** 🎉  
If you want, create a reusable config (`--save-config`) and run future datasets with just one command.