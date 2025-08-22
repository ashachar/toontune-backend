import os, platform, sys
print(f"[LAMBDA_MANIFEST] Python {sys.version.split()[0]} on {platform.system()} {platform.release()} arch={platform.machine()}")
print(f"[LAMBDA_MANIFEST] AWS_EXECUTION_ENV={os.environ.get('AWS_EXECUTION_ENV')}")
print(f"[LAMBDA_MANIFEST] LAMBDA_TASK_ROOT={os.environ.get('LAMBDA_TASK_ROOT')}")
print(f"[LAMBDA_MANIFEST] PATH={os.environ.get('PATH')}")
print(f"[LAMBDA_MANIFEST] REMBG_SESSION(default)=u2net (init happens in code)")
