from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

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

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    data_path: Path
    STATUS_FILE: Path
    all_schema: dict  # Loaded from schema.yaml COLUMNS


@dataclass(frozen=True)
class NonRecursiveTrainerConfig:
    root_dir: Path
    processed_data_dir: Path
    models_dir: Path
    first_day: int
    cv: str
    early_stopping_rounds: int
    stores: List[str]
    cats: List[str]
    depts: List[str]
    lgb_params: Dict
    validation: Dict
    grid2_colnm: List[str]
    grid3_colnm: List[str]
    lag_colnm: List[str]
    remove_feature_by_level: Dict
    mean_enc_by_level: Dict


@dataclass(frozen=True)
class RecursiveTrainerConfig:
    root_dir: Path
    processed_data_dir: Path
    models_dir: Path
    end_train: int
    p_horizon: int
    ver: str
    early_stopping_rounds: int
    stores: List[str]
    cats: List[str]
    depts: List[str]
    start_train_by_level: Dict
    remove_feature_by_level: Dict
    mean_enc_by_level: Dict
    lgb_params_store: Dict
    lgb_params_default: Dict


@dataclass(frozen=True)
class RecursivePredictorConfig:
    processed_data_dir: Path
    models_dir: Path
    submission_dir: Path
    sample_submission_path: Path
    end_train: int
    p_horizon: int
    ver: str
    stores: List[str]
    cats: List[str]
    depts: List[str]
    remove_feature_by_level: Dict
    mean_enc_by_level: Dict
    rols_split: List[List[int]]

@dataclass(frozen=True)
class NonRecursivePredictorConfig:
    processed_data_dir: Path
    models_dir: Path
    submission_dir: Path
    log_dir: Path
    sample_submission_path: Path
    cv: str
    first_day: int
    end_train: int
    p_horizon: int
    stores: List[str]
    cats: List[str]
    depts: List[str]
    grid2_colnm: List[str]
    grid3_colnm: List[str]
    lag_colnm: List[str]
    remove_feature_by_level: Dict
    mean_enc_by_level: Dict


@dataclass(frozen=True)
class EnsembleConfig:
    submission_dir: Path
    sample_submission_path: Path
    submission_files: List[str]
    final_submission_filename: str
    eval_row_offset: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    sales_train_path: Path
    sales_test_path: Path
    sell_prices_path: Path
    calendar_path: Path
    final_submission_path: Path
    metric_file_name: Path
    end_train: int
    p_horizon: int