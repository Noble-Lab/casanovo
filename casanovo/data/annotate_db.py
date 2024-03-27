"""Methods used to annotate an .mgf so that it can be used by Casanovo-DB"""

from pathlib import Path
from typing import Optional, Tuple


def annotate_mgf(peak_path: str, tide_path: str, output: Optional[str]):
    print(peak_path, tide_path, output)
    ## TODO
