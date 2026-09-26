from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")
SCHEMA_FILE_PATH = Path("schema.yaml")

LEVEL_KEYS = {
    "Level1":  [],
    "Level2":  ["state_id"],
    "Level3":  ["store_id"],
    "Level4":  ["cat_id"],
    "Level5":  ["dept_id"],
    "Level6":  ["state_id", "cat_id"],
    "Level7":  ["state_id", "dept_id"],
    "Level8":  ["store_id", "cat_id"],
    "Level9":  ["store_id", "dept_id"],
    "Level10": ["item_id"],
    "Level11": ["state_id", "item_id"],
    "Level12": ["item_id", "store_id"],
}