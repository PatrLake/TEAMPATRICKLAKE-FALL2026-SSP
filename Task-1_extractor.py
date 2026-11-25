from pathlib import Path
from pypdf import PdfReader
import sys
import yaml

def is_valid_input(filepath1: str, filepath2: str) -> bool:
    for filepath in (filepath1, filepath2):
        try:
            path = Path(filepath)
            if not path.is_file() or path.stat().st_size == 0:
                return False
            with path.open("rb") as f:
                if not f.read(5).startswith(b"%PDF-"):
                    return False
            reader = PdfReader(path, strict=False)
            if reader.is_encrypted or len(reader.pages) == 0:
                return False
            _ = reader.pages[0]
        except (Exception):
            return False
    return True

def construct_zero_shot_prompt(filepath1: str, filepath2: str) -> str:
    pass

def construct_few_shot_prompt(filepath1: str, filepath2: str) -> str:
    pass

def construct_chain_of_thought_prompt(filepath1: str, filepath2: str) -> str:
    pass

def identify_KDEs(prompt: str) -> dict:
    pass

def create_TXT_file(prompts: list, outputs: list) -> None:
    pass

if __name__ == "__main__":
    filepath1 = sys.argv[1]
    filepath2 = sys.argv[2]