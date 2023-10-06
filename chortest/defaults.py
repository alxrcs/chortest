from pathlib import Path
import os.path
import shutil

current_path = Path(__file__)
target_folder_name = "chorgram"

def check_if_chorgram_is_a_parent_folder():
    result_path = current_path
    for part in reversed(current_path.parts):
        if part == target_folder_name:
            return result_path
        result_path = result_path.parent
    return None

def check_if_chorgram_is_a_subfolder():
    for part in current_path.parents:
        possible_path = os.path.join(part, target_folder_name)
        if os.path.exists(possible_path):
            return Path(possible_path)
    return None

def check_if_chorgram_is_in_path():
    exe = shutil.which("chorgram")
    if exe is not None:
        return Path(exe).parent
    else:
        raise FileNotFoundError("The chorgram executable was not found in the PATH. Please make sure that chorgram is installed and accessible from PATH.")

DEFAULT_CHORGRAM_BASE_PATH: Path = check_if_chorgram_is_a_parent_folder() or check_if_chorgram_is_a_subfolder() or check_if_chorgram_is_in_path()
CHORTEST_BASE_PATH: Path = current_path.parent