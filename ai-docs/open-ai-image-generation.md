# 🐍 Claude Task: Generate a **high-quality colorful doodle of a kid** with a transparent background using **OpenAI `image-1` model** (not DALL·E).

Follow exactly. This is **Python-only** and **text → image only** (no editing).

---

## 📁 Project Layout

```
openai-image-sandbox/
├── README.md
├── out/
└── python/
    └── generate.py
```

---

## 🔐 Prerequisites

- OpenAI API key available as `OPENAI_API_KEY` in your shell.
- Python **3.9+** installed.

---

## 🧾 Create `README.md`

**File: `openai-image-sandbox/README.md`**

```md
# OpenAI Image Sandbox (Python, image-1)

This sandbox generates a **colorful doodle of a kid** with a transparent background using the `image-1` model.

## Prerequisites
- Set your OpenAI key:
  - macOS/Linux:
    ```bash
    export OPENAI_API_KEY="sk-..."
    ```
  - Windows (PowerShell):
    ```powershell
    setx OPENAI_API_KEY "sk-..."
    $env:OPENAI_API_KEY="sk-..."
    ```

## Python Quickstart
```bash
cd python
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install openai==1.40.0 python-dotenv==1.0.1
python generate.py          # text -> image with transparent background
```

## Background removal (if needed)

If transparency is not returned:

```bash
pip install rembg
rembg i ../out/kid.png ../out/kid_transparent.png
```
```

---

## 🐍 Python Code

### Create `openai-image-sandbox/python/generate.py`

```python
# Generate a high-quality colorful doodle kid with transparent background using OpenAI image-1
import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Export it in your shell and try again.")

    client = OpenAI(api_key=api_key)

    prompt = (
        "high quality colorful doodle of a smiling kid, thick black outline, simple hand-drawn shapes, "
        "flat bright crayon colors, playful proportions, minimal details, whimsical cartoon doodle style, "
        "clean edges, no shading, transparent background"
    )

    print("▶️ Generating high-quality doodle kid with image-1...")
    resp = client.images.generate(
        model="image-1",
        prompt=prompt,
        size="1024x1024",
        background="transparent"
    )

    b64 = resp.data[0].b64_json
    out_dir = os.path.join("..", "out")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "kid.png")
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"✅ Saved {out_path}")

if __name__ == "__main__":
    main()
```

---

## ▶️ How to Run

```bash
# 1) Create and activate a virtual environment
cd openai-image-sandbox/python
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install openai==1.40.0 python-dotenv==1.0.1

# 3) Set your API key
export OPENAI_API_KEY="sk-..."   # macOS/Linux
# or
setx OPENAI_API_KEY "sk-..."     # Windows

# 4) Generate image
python generate.py    # outputs ../out/kid.png
```

---

## 🛠️ Notes

* **Transparent background** is requested via `background="transparent"`.
  If your region/model returns opaque backgrounds, run `rembg` as shown above.
* The `image-1` model is newer and higher fidelity than DALL·E 2 for certain styles, giving better line quality for doodles.
* Adjust prompt wording for more/less detail (e.g., "minimalist" vs "highly detailed").