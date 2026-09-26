from SellWise.config.configuration import ConfigurationManager
from SellWise.components.nonrecursive_predicting import NonRecursivePredictor
from SellWise import logger


STAGE_NAME = "Non-Recursive Prediction stage"

class NonRecursivePredictionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        nonrecursive_predictor_config = config.get_nonrecursive_predictor_config()
        predictor = NonRecursivePredictor(config=nonrecursive_predictor_config)
        predictor.run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        pipeline = NonRecursivePredictionPipeline()
        pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed!<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e