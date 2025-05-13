import sys
import os
from pathlib import Path

# Add the src directory to the Python path
root_dir = Path(__file__).parent.parent
src_dir = root_dir / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))
else:
    sys.path.insert(0, str(root_dir))