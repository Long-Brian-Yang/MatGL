import ase.db
from ase.io import read

bulk = read("OUTCAR", index=":")
dbname = "bulk.db"

db = ase.db.connect(dbname)
for i in bulk:
    db.write(i)

print(f"Number of frames: {len(bulk)}")
print(f"Data stored to {dbname}")
