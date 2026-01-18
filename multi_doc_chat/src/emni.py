from pathlib import Path

d = Path("darling/LOOO")
d.mkdir(parents=True,exist_ok=True)

k = Path("LOOO")
print(k.resolve())