from SellWise import logger
from SellWise.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from SellWise.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline
from SellWise.pipeline.stage_03_data_validation import DataValidationPipeline
from SellWise.pipeline.stage_04_nonrecursive_training import NonRecursiveTrainingPipeline
from SellWise.pipeline.stage_05_recursive_training import RecursiveTrainingPipeline
from SellWise.pipeline.stage_06_recursive_prediction import RecursivePredictionPipeline
from SellWise.pipeline.stage_07_nonrecursive_prediction import NonRecursivePredictionPipeline
from SellWise.pipeline.stage_08_ensemble import EnsemblePredictionPipeline
from SellWise.pipeline.stage_09_model_evaluation import ModelEvaluationPipeline
# logger.info("Welcome to our SellWise package")


STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Preprocessing Stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_preprocessing = DataPreprocessingPipeline()
   data_preprocessing.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Validation Stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_validation = DataValidationPipeline()
   data_validation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Non-Recursive Training Stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   nonrecursive_training = NonRecursiveTrainingPipeline()
   nonrecursive_training.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Recursive Training Stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   recursive_training = RecursiveTrainingPipeline()
   recursive_training.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Recursive Prediction Stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   recursive_prediction = RecursivePredictionPipeline()
   recursive_prediction.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Non-Recursive Prediction Stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   nonrecursive_prediction = NonRecursivePredictionPipeline()
   nonrecursive_prediction.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x") 
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Ensemble Prediction Stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   ensemble_prediction = EnsemblePredictionPipeline()
   ensemble_prediction.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x") 
except Exception as e:
        logger.exception(e)
        raise e

# =========================================================
# STAGE 09 - MODEL EVALUATION
# =========================================================

STAGE_NAME = "Model Evaluation Stage"

try:

    logger.info(
        f">>>>>> stage "
        f"{STAGE_NAME} started <<<<<<"
    )

    model_evaluation = (
        ModelEvaluationPipeline()
    )

    model_evaluation.main()

    logger.info(
        f">>>>>> stage "
        f"{STAGE_NAME} completed <<<<<<"
        "\n\nx==========x"
    )

except Exception as e:

    logger.exception(e)

    raise e
