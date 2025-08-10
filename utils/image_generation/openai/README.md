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
cd utils/image_generation/openai
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install openai python-dotenv
python generate.py          # text -> image with transparent background
```

## Background removal (if needed)

If transparency is not returned:

```bash
pip install rembg
rembg i out/kid.png out/kid_transparent.png
```

## Files
- `generate.py` - Generate high-quality colorful doodle kid from text prompt
- `out/` - Output directory for generated images