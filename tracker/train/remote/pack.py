"""pack.py — the files asked for, into one tar to download in a single transfer.

    python3 pack.py OUT.tar PATH...

Each file is stored under its path relative to the home directory; paths that
do not exist or lie outside home are skipped. Prints how many were packed.
"""
import os
import sys
import tarfile

home = os.path.realpath(os.path.expanduser("~"))
out, paths = sys.argv[1], sys.argv[2:]
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
packed = 0
with tarfile.open(out, "w") as tar:
    for path in paths:
        real = os.path.realpath(os.path.expanduser(path))
        if not real.startswith(home + os.sep) or not os.path.isfile(real):
            continue
        tar.add(real, arcname=os.path.relpath(real, home).replace(os.sep, "/"))
        packed += 1
print(f"PACKED {packed}")
