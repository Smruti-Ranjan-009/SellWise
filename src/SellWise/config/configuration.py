from SellWise.constants import *
from SellWise.utils.common import read_yaml, create_directories
from SellWise.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig)

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