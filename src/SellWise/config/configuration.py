from SellWise.constants import *
from SellWise.utils.common import read_yaml, create_directories
from SellWise.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig, DataValidationConfig, NonRecursiveTrainerConfig, RecursiveTrainerConfig, RecursivePredictorConfig, NonRecursivePredictorConfig, EnsembleConfig, ModelEvaluationConfig)

class ConfigurationManager:
    def __init__(
        self, 
        config_filepath=CONFIG_FILE_PATH, 
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        create_directories([(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            calendar_path=config.calendar_path,
            sell_prices_path=config.sell_prices_path,
            sales_path=config.sales_path
        )
        
        return data_ingestion_config

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
            config = self.config.data_preprocessing
            ingestion_config = self.config.data_ingestion
    
            create_directories([config.root_dir])
    
            data_preprocessing_config = DataPreprocessingConfig(
                root_dir=config.root_dir,
                calendar_path=ingestion_config.calendar_path,
                sell_prices_path=ingestion_config.sell_prices_path,
                sales_path=ingestion_config.sales_path,
                grid_part_1=config.grid_part_1,
                grid_part_2=config.grid_part_2,
                grid_part_3=config.grid_part_3,
                lags_df_28=config.lags_df_28,
                mean_encoding_df=config.mean_encoding_df
            )
    
            return data_preprocessing_config

    def get_data_validation_config(self) -> DataValidationConfig:
            config = self.config.data_validation
            schema = self.schema.COLUMNS  # Loads all expected features from schema.yaml
    
            create_directories([config.root_dir])
    
            data_validation_config = DataValidationConfig(
                root_dir=Path(config.root_dir),
                data_path=Path(config.data_path),
                STATUS_FILE=Path(config.STATUS_FILE),
                all_schema=schema
            )
    
            return data_validation_config

    def get_nonrecursive_trainer_config(self) -> NonRecursiveTrainerConfig:
            config = self.config.fleet_trainer
            params = self.params.NONRECURSIVE_TRAINING
            lgb_params = self.params.LIGHTGBM
    
            create_directories([config.models_dir])
    
            nonrecursive_trainer_config = NonRecursiveTrainerConfig(
                root_dir=config.root_dir,
                processed_data_dir=config.processed_data_dir,
                models_dir=config.models_dir,
                first_day=params.first_day,
                cv=params.cv,
                early_stopping_rounds=params.early_stopping_rounds,
                stores=list(params.stores),
                cats=list(params.cats),
                depts=list(params.depts),
                lgb_params=dict(lgb_params),
                validation={k: list(v) for k, v in params.validation.items()},
                grid2_colnm=list(params.grid2_colnm),
                grid3_colnm=list(params.grid3_colnm),
                lag_colnm=list(params.lag_colnm),
                remove_feature_by_level={k: list(v) for k, v in params.remove_feature_by_level.items()},
                mean_enc_by_level={k: list(v) for k, v in params.mean_enc_by_level.items()}
            )

            return nonrecursive_trainer_config

    def get_recursive_trainer_config(self) -> RecursiveTrainerConfig:
            config = self.config.fleet_trainer
            params = self.params.RECURSIVE_TRAINING
    
            create_directories([config.models_dir])
    
            recursive_trainer_config = RecursiveTrainerConfig(
                root_dir=config.root_dir,
                processed_data_dir=config.processed_data_dir,
                models_dir=config.models_dir,
                end_train=params.end_train,
                p_horizon=params.p_horizon,
                ver=params.ver,
                early_stopping_rounds=params.early_stopping_rounds,
                stores=list(params.stores),
                cats=list(params.cats),
                depts=list(params.depts),
                start_train_by_level=dict(params.start_train_by_level),
                remove_feature_by_level={k: list(v) for k, v in params.remove_feature_by_level.items()},
                mean_enc_by_level={k: list(v) for k, v in params.mean_enc_by_level.items()},
                lgb_params_store=dict(params.lgb_params_store),
                lgb_params_default=dict(params.lgb_params_default)
            )

            return recursive_trainer_config

    def get_recursive_predictor_config(self) -> RecursivePredictorConfig:
            fleet_config = self.config.fleet_trainer
            pred_config = self.config.predictor
            ingestion_config = self.config.data_ingestion
            params = self.params.RECURSIVE_TRAINING
    
            create_directories([pred_config.submission_dir])
    
            recursive_predictor_config = RecursivePredictorConfig(
                processed_data_dir=fleet_config.processed_data_dir,
                models_dir=fleet_config.models_dir,
                submission_dir=pred_config.submission_dir,
                sample_submission_path=ingestion_config.sample_submission_path,
                end_train=params.end_train,
                p_horizon=params.p_horizon,
                ver=params.ver,
                stores=list(params.stores),
                cats=list(params.cats),
                depts=list(params.depts),
                remove_feature_by_level={k: list(v) for k, v in params.remove_feature_by_level.items()},
                mean_enc_by_level={k: list(v) for k, v in params.mean_enc_by_level.items()},
                rols_split=[list(pair) for pair in params.rols_split]
            )

            return recursive_predictor_config

    def get_nonrecursive_predictor_config(self) -> NonRecursivePredictorConfig:
            fleet_config = self.config.fleet_trainer
            pred_config = self.config.predictor
            ingestion_config = self.config.data_ingestion
            params = self.params.NONRECURSIVE_TRAINING
            # A1's validation['private'] = [1941, 1969]: predict d in (start, end]
            cv_start, cv_end = params.validation[params.cv]
    
            # per-group prediction CSVs (A1's "4. logs/"); final files go straight
            # into submission_dir, next to the recursive predictor's outputs
            log_dir = Path(pred_config.submission_dir) / "nonrecursive_logs"
            create_directories([pred_config.submission_dir, log_dir])
    
            nonrecursive_predictor_config = NonRecursivePredictorConfig(
                processed_data_dir=fleet_config.processed_data_dir,
                models_dir=fleet_config.models_dir,
                submission_dir=pred_config.submission_dir,
                log_dir=log_dir,
                sample_submission_path=ingestion_config.sample_submission_path,
                cv=params.cv,
                first_day=params.first_day,
                end_train=cv_start,
                p_horizon=cv_end - cv_start,
                stores=list(params.stores),
                cats=list(params.cats),
                depts=list(params.depts),
                grid2_colnm=list(params.grid2_colnm),
                grid3_colnm=list(params.grid3_colnm),
                lag_colnm=list(params.lag_colnm),
                remove_feature_by_level={k: list(v) for k, v in params.remove_feature_by_level.items()},
                mean_enc_by_level={k: list(v) for k, v in params.mean_enc_by_level.items()}
            )

            return nonrecursive_predictor_config

    def get_ensemble_config(self) -> EnsembleConfig:
            pred_config = self.config.predictor
            ingestion_config = self.config.data_ingestion
    
            create_directories([pred_config.submission_dir])
    
            # List of the 6 individual model prediction submission files
            submission_files = [
                "submission_recursive_store.csv",
                "submission_recursive_store_cat.csv",
                "submission_recursive_store_dept.csv",
                "submission_nonrecursive_store.csv",
                "submission_nonrecursive_store_cat.csv",
                "submission_nonrecursive_store_dept.csv"
            ]
    
            ensemble_config = EnsembleConfig(
                submission_dir=Path(pred_config.submission_dir),
                sample_submission_path=Path(ingestion_config.sample_submission_path),
                submission_files=submission_files,
                final_submission_filename="submission_final.csv",
                eval_row_offset=30490  # Evaluation row index split from original notebook
            )

            return ensemble_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
            eval_config = self.config.model_evaluation
            ingestion_config = self.config.data_ingestion
            recursive_params = self.params.RECURSIVE_TRAINING
    
            create_directories([eval_config.root_dir])
    
            model_evaluation_config = ModelEvaluationConfig(
                root_dir=eval_config.root_dir,
                sales_train_path=ingestion_config.sales_path,
                sales_test_path=eval_config.sales_test_evaluation_path,
                sell_prices_path=ingestion_config.sell_prices_path,
                calendar_path=ingestion_config.calendar_path,
                final_submission_path=eval_config.final_submission_path,
                metric_file_name=eval_config.metric_file_name,
                end_train=recursive_params.end_train,
                p_horizon=recursive_params.p_horizon
            )

            return model_evaluation_config