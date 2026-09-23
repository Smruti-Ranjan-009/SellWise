import os
from SellWise import logger
from SellWise.entity.config_entity import DataValidationConfig
import pandas as pd

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_and_columns(self) -> bool:
        try:
            validation_status = True
            
            # 1. Check artifact existence & non-zero size
            required_files = [
                "grid_part_1.pkl",
                "grid_part_2.pkl",
                "grid_part_3.pkl",
                "lags_df_28.pkl",
                "mean_encoding_df.pkl"
            ]

            existing_files = os.listdir(self.config.data_path)

            for file in required_files:
                file_full_path = os.path.join(self.config.data_path, file)
                if file not in existing_files or os.path.getsize(file_full_path) == 0:
                    validation_status = False
                    logger.error(f"Missing or empty artifact: {file}")
                    break

            # 2. Check column schema on sample artifact (grid_part_1.pkl)
            if validation_status:
                sample_data = pd.read_pickle(os.path.join(self.config.data_path, "grid_part_1.pkl"))
                all_cols = list(sample_data.columns)
                expected_schema = self.config.all_schema.keys()

                for col in all_cols:
                    if col not in expected_schema:
                        validation_status = False
                        logger.error(f"Unexpected column found: {col}")
                        break

            # 3. Write overall status to status.txt
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e