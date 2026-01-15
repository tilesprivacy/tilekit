from pathlib import Path
import os

PORT = 6969


def get_model_and_prompt():
    """Get model ID and prompt based on TILES_TRACK."""
    track = os.environ.get("TILES_TRACK", "regular").lower()
    
    if track == "insider":
        model_id = "driaforall/mem-agent"
        prompt_file = "system_prompt.txt"
    else:
        model_id = "mlx-community/gpt-oss-20b-MXFP4-Q4"
        prompt_file = "gpt_oss_prompt.txt"
    
    prompt_path = Path(__file__).parent / prompt_file
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    
    return model_id, prompt


# Module-level defaults
TILES_TRACK = os.environ.get("TILES_TRACK", "regular").lower()
MODEL_ID, SYSTEM_PROMPT = get_model_and_prompt()
MEMORY_PATH = os.path.expanduser("~") + "/tiles_memory"
