from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    calendar_path: Path
    sell_prices_path: Path
    sales_path: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    calendar_path: Path
    sell_prices_path: Path
    sales_path: Path
    grid_part_1: Path
    grid_part_2: Path
    grid_part_3: Path
    lags_df_28: Path
    mean_encoding_df: Path


from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_path: Path
    STATUS_FILE: Path
    all_schema: dict  # Loaded from schema.yaml COLUMNS
