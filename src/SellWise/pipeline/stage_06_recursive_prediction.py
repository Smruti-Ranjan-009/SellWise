from SellWise.config.configuration import ConfigurationManager
from SellWise.components.recursive_predicting import RecursivePredictor
from SellWise import logger

STAGE_NAME = "Recursive Prediction stage"

class RecursivePredictionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        recursive_predictor_config = config.get_recursive_predictor_config()
        predictor = RecursivePredictor(config=recursive_predictor_config)
        predictor.run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        pipeline = RecursivePredictionPipeline()
        pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed!<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e