import os
import shutil

SAFE_TRASH = [
    ".venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints"
]

def safe_remove(path):
    if os.path.exists(path):
        print(f"Removing: {path}")
        shutil.rmtree(path, ignore_errors=True)

def remove_pyc(root="."):
    for r, dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".pyc"):
                full = os.path.join(r, f)
                print(f"Deleting PYC: {full}")
                try:
                    os.remove(full)
                except:
                    pass

if __name__ == "__main__":
    print("Safe cleanup started...")
    for p in SAFE_TRASH:
        safe_remove(p)
    remove_pyc()
    print("Cleanup complete. All source code untouched.")
