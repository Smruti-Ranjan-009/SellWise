from SellWise.config.configuration import ConfigurationManager
from SellWise.components.nonrecursive_training import NonRecursiveTrainer
from SellWise import logger

STAGE_NAME = "Non-Recursive Training stage"

class NonRecursiveTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        nonrecursive_trainer_config = config.get_nonrecursive_trainer_config()
        trainer = NonRecursiveTrainer(config=nonrecursive_trainer_config)
        trainer.run()



if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        pipeline = NonRecursiveTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed!<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e