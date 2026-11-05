import importlib

libraries = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "sklearn",
    "scipy",
    "joblib"
]

print("=" * 45)
print("REQUIRED LIBRARIES CHECK")
print("=" * 45)

for library in libraries:
    try:
        importlib.import_module(library)
        print(f"[INSTALLED] {library}")
    except ImportError:
        print(f"[MISSING] {library}")

print("=" * 45)