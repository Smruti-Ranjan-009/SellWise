from SellWise.config.configuration import ConfigurationManager
from SellWise.components.recursive_training import RecursiveTrainer
from SellWise import logger

STAGE_NAME = "Recursive Training stage"

class RecursiveTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        recursive_trainer_config = config.get_recursive_trainer_config()
        trainer = RecursiveTrainer(config=recursive_trainer_config)
        trainer.run()


if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        pipeline = RecursiveTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed!<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e