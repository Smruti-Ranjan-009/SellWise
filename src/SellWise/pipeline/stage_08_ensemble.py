from SellWise.config.configuration import ConfigurationManager
from SellWise.components.ensemble import EnsemblePredictor
from SellWise import logger


STAGE_NAME = "Ensemble Prediction stage"

class EnsemblePredictionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        ensemble_config = config.get_ensemble_config()
        predictor = EnsemblePredictor(config=ensemble_config)
        final_submission = predictor.run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        pipeline = EnsemblePredictionPipeline()
        pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed!<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e