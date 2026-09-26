import os

from dotenv import load_dotenv

from SellWise import logger


def load_environment() -> None:
    """
    Load environment variables from the project's .env file.

    The .env file must not be committed to Git.
    """

    load_dotenv()

    logger.info("Environment variables loaded")


def validate_mlflow_environment() -> None:
    """
    Ensure all environment variables required for
    MLflow/DagsHub tracking are available.
    """

    required_variables = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
    ]

    missing_variables = [
        variable
        for variable in required_variables
        if not os.getenv(variable)
    ]

    if missing_variables:
        raise EnvironmentError(
            "Missing required MLflow environment variables: "
            + ", ".join(missing_variables)
        )

    logger.info(
        "MLflow environment variables validated successfully"
    )


def get_mlflow_tracking_uri() -> str:
    """
    Return the MLflow tracking URI from the environment.
    """

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        raise EnvironmentError(
            "MLFLOW_TRACKING_URI is not set"
        )

    return tracking_uri