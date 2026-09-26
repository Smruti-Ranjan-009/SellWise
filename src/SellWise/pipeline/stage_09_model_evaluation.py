from SellWise import logger

from SellWise.config.configuration import (
    ConfigurationManager
)

from SellWise.components.model_evaluation import (
    ModelEvaluation
)

from SellWise.utils.env_utils import (
    load_environment,
    validate_mlflow_environment,
)


STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationPipeline:

    def __init__(self):
        pass

    def main(self):

        # ---------------------------------------------
        # Load .env
        # ---------------------------------------------

        load_environment()

        # ---------------------------------------------
        # Validate MLflow/DagsHub credentials
        # ---------------------------------------------

        validate_mlflow_environment()

        # ---------------------------------------------
        # Load component configuration
        # ---------------------------------------------

        config = ConfigurationManager()

        model_evaluation_config = (
            config.get_model_evaluation_config()
        )

        # ---------------------------------------------
        # Run evaluation
        # ---------------------------------------------

        model_evaluation = ModelEvaluation(
            config=model_evaluation_config
        )

        scores = (
            model_evaluation.log_into_mlflow()
        )

        return scores


if __name__ == "__main__":

    try:

        logger.info(
            f">>>>>> stage "
            f"{STAGE_NAME} started <<<<<<"
        )

        pipeline = (
            ModelEvaluationPipeline()
        )

        pipeline.main()

        logger.info(
            f">>>>>> stage "
            f"{STAGE_NAME} completed <<<<<<"
            "\n\nx==========x"
        )

    except Exception as e:

        logger.exception(e)

        raise e