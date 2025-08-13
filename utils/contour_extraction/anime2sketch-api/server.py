import os
import uuid
import shutil
import subprocess
import logging
import time
import asyncio
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

BASE = os.path.abspath(os.path.dirname(__file__))
REPO = os.path.join(BASE, "Anime2Sketch")
TMP = os.path.join(BASE, "tmp")
os.makedirs(TMP, exist_ok=True)

def repo_ready():
    if not os.path.isdir(REPO):
        return False, f"Repository missing at {REPO}"
    wd = os.path.join(REPO, "weights")
    if not os.path.isdir(wd):
        return False, f"Weights directory missing at {wd}"
    weight_files = os.listdir(wd)
    if not weight_files:
        return False, f"No weight files found in {wd}"
    return True, f"ok - found weights: {weight_files}"

@app.get("/")
def root():
    return {"message": "Anime2Sketch API", "status": "ready"}

@app.get("/healthz")
def health():
    ok, msg = repo_ready()
    return {"status": "ok" if ok else "not_ready", "msg": msg}

@app.get("/debug")
def debug():
    """Debug endpoint to check environment"""
    repo_path = os.path.join(BASE, "Anime2Sketch")
    weights_path = os.path.join(repo_path, "weights")
    
    debug_info = {
        "base": BASE,
        "repo_exists": os.path.exists(repo_path),
        "weights_exists": os.path.exists(weights_path),
        "weights_files": os.listdir(weights_path) if os.path.exists(weights_path) else [],
        "repo_files": os.listdir(repo_path)[:20] if os.path.exists(repo_path) else [],
        "tmp_exists": os.path.exists(TMP),
        "python_path": subprocess.run(["which", "python"], capture_output=True, text=True).stdout.strip()
    }
    return debug_info

@app.post("/infer")
async def infer(file: UploadFile = File(...), load_size: int = 512, debug_mode: bool = False):
    ok, msg = repo_ready()
    if not ok:
        return JSONResponse({"error": msg}, status_code=500)
    
    rid = str(uuid.uuid4())
    idir = os.path.join(TMP, f"in_{rid}")
    odir = os.path.join(TMP, f"out_{rid}")
    os.makedirs(idir)
    os.makedirs(odir)
    
    # Save input file
    input_path = os.path.join(idir, file.filename)
    with open(input_path, "wb") as f:
        f.write(await file.read())
    
    # Expected output path (Anime2Sketch uses same filename)
    expected_output = os.path.join(odir, file.filename)
    
    cleanup_needed = True
    
    try:
        # Run the model
        logger.info(f"Running model with input: {input_path}, output_dir: {odir}")
        result = subprocess.run([
            "python", "test.py", 
            "--dataroot", idir,
            "--load_size", str(load_size), 
            "--output_dir", odir
        ], cwd=REPO, capture_output=True, text=True, timeout=30)
        
        # If debug mode, return the subprocess output
        if debug_mode:
            return JSONResponse({
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "output_files": os.listdir(odir) if os.path.exists(odir) else [],
                "expected_output": expected_output,
                "expected_exists": os.path.exists(expected_output)
            })
        
        # Log output for debugging
        if result.stdout:
            logger.info(f"Model stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"Model stderr: {result.stderr}")
        
        # Check if model ran successfully
        if result.returncode != 0:
            return JSONResponse({
                "error": f"Model failed with code {result.returncode}",
                "stderr": result.stderr,
                "stdout": result.stdout
            }, status_code=500)
        
        # Small delay to ensure file is written
        await asyncio.sleep(0.1)
        
        # Check if expected output exists
        if os.path.exists(expected_output):
            # Read the file content
            with open(expected_output, 'rb') as f:
                content = f.read()
            
            # Return the content as a Response
            output_name = os.path.splitext(file.filename)[0] + "_sketch.jpg"
            headers = {
                "Content-Disposition": f'attachment; filename="{output_name}"'
            }
            
            # Now cleanup is safe since we've read the file
            shutil.rmtree(idir, ignore_errors=True)
            shutil.rmtree(odir, ignore_errors=True)
            cleanup_needed = False
            
            return Response(content=content, media_type="image/jpeg", headers=headers)
        
        # If not found, list what's actually in the output directory
        actual_files = os.listdir(odir) if os.path.exists(odir) else []
        
        # Try to find any image file
        for f in actual_files:
            full_path = os.path.join(odir, f)
            if os.path.isfile(full_path) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                logger.info(f"Found output file: {full_path}")
                
                # Read the file content
                with open(full_path, 'rb') as file_obj:
                    content = file_obj.read()
                
                output_name = os.path.splitext(file.filename)[0] + "_sketch" + os.path.splitext(f)[1]
                headers = {
                    "Content-Disposition": f'attachment; filename="{output_name}"'
                }
                
                # Cleanup after reading
                shutil.rmtree(idir, ignore_errors=True)
                shutil.rmtree(odir, ignore_errors=True)
                cleanup_needed = False
                
                return Response(content=content, media_type="image/jpeg", headers=headers)
        
        # No output found
        return JSONResponse({
            "error": "No output generated",
            "expected": expected_output,
            "actual_files": actual_files,
            "stdout": result.stdout,
            "stderr": result.stderr
        }, status_code=500)
        
    except subprocess.TimeoutExpired:
        return JSONResponse({"error": "Model timeout after 30 seconds"}, status_code=500)
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        # Cleanup only if not already done
        if cleanup_needed and not debug_mode:
            shutil.rmtree(idir, ignore_errors=True)
            shutil.rmtree(odir, ignore_errors=True)