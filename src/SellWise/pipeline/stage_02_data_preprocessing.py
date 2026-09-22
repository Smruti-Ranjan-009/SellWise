from SellWise.config.configuration import ConfigurationManager  
from SellWise.components.data_preprocessing import DataPreprocessing
from SellWise import logger


STAGE_NAME = "Data Preprocessing Stage"

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        preprocessing_config = config.get_data_preprocessing_config()
        data_preprocessing = DataPreprocessing(config=preprocessing_config)
        
        data_preprocessing.create_grid_part_1()
        data_preprocessing.create_grid_part_2()
        data_preprocessing.create_grid_part_3()
        data_preprocessing.finalize_part_1()
        data_preprocessing.create_lags()
        data_preprocessing.create_mean_encodings()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e