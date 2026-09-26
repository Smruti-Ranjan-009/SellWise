from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
from SellWise import logger
from SellWise.constants import LEVEL_KEYS
from SellWise.entity.config_entity import ModelEvaluationConfig
from SellWise.utils.common import save_json
from SellWise.utils.env_utils import get_mlflow_tracking_uri



class ModelEvaluation:
    """
    Computes the official-style M5 12-level hierarchical
    Weighted Root Mean Squared Scaled Error (WRMSSE).

    Bottom-level item x store forecasts are aggregated into
    all 12 M5 hierarchy levels.

    For each level:

        1. Sales are aggregated.
        2. Forecasts are aggregated.
        3. RMSSE is calculated.
        4. Dollar-sales weights are applied.

    The final WRMSSE is the mean of the 12 level scores.
    """

    def __init__(
        self,
        config: ModelEvaluationConfig
    ):
        self.config = config

    # =====================================================
    # DATA LOADING
    # =====================================================

    def load_data(self) -> None:

        logger.info(
            "Loading model evaluation datasets"
        )

        self.sales_train = pd.read_csv(
            self.config.sales_train_path
        )

        self.sales_test = pd.read_csv(
            self.config.sales_test_path
        )

        self.sell_prices = pd.read_csv(
            self.config.sell_prices_path
        )

        self.calendar = pd.read_csv(
            self.config.calendar_path
        )

        self.submission = pd.read_csv(
            self.config.final_submission_path
        )

        logger.info(
            "Model evaluation datasets loaded successfully"
        )

    # =====================================================
    # BUILD EVALUATION FRAMES
    # =====================================================

    def _build_frames(self):
        """
        Align training history, actual evaluation sales,
        and model forecasts into identical bottom-level
        item x store ordering.
        """

        meta_cols = [
            "item_id",
            "dept_id",
            "cat_id",
            "store_id",
            "state_id",
        ]

        # -------------------------------------------------
        # Training history
        # -------------------------------------------------

        train_day_cols = [
            col
            for col in self.sales_train.columns
            if col.startswith("d_")
        ]

        meta = (
            self.sales_train[
                meta_cols
            ]
            .reset_index(drop=True)
        )

        train_wide = (
            self.sales_train[
                train_day_cols
            ]
            .reset_index(drop=True)
        )

        # -------------------------------------------------
        # Actual evaluation values
        # -------------------------------------------------

        test_wide = meta.merge(
            self.sales_test,
            on=meta_cols,
            how="left"
        )

        test_day_cols = [
            col
            for col in self.sales_test.columns
            if col.startswith("d_")
        ]

        test_wide = (
            test_wide[
                test_day_cols
            ]
            .reset_index(drop=True)
        )

        # -------------------------------------------------
        # Forecast values
        # -------------------------------------------------

        id_map = self.sales_train[
            ["id"] + meta_cols
        ]

        forecast_full = id_map.merge(
            self.submission,
            on="id",
            how="left"
        )

        forecast_cols = [
            col
            for col in self.submission.columns
            if col.startswith("F")
        ]

        forecast_wide = (
            forecast_full[
                forecast_cols
            ]
            .reset_index(drop=True)
        )

        # F1...F28 must correspond to test d_* columns
        if len(forecast_cols) != len(test_day_cols):
            raise ValueError(
                "Forecast horizon and evaluation horizon "
                "do not match. "
                f"Forecast columns: {len(forecast_cols)}, "
                f"Test columns: {len(test_day_cols)}"
            )

        forecast_wide.columns = test_day_cols

        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        expected_rows = len(meta)

        if not (
            len(train_wide)
            == len(test_wide)
            == len(forecast_wide)
            == expected_rows
        ):
            raise ValueError(
                "Row count mismatch after evaluation "
                "alignment. Check submission IDs and "
                "(item_id, store_id) mappings."
            )

        if test_wide.isna().any().any():
            raise ValueError(
                "Missing actual sales values found in "
                "evaluation dataset."
            )

        if forecast_wide.isna().any().any():
            raise ValueError(
                "Missing forecast values found in "
                "final submission."
            )

        logger.info(
            "Evaluation frames aligned successfully: "
            f"{expected_rows} bottom-level series"
        )

        return (
            meta,
            train_wide,
            test_wide,
            forecast_wide,
        )

    # =====================================================
    # DOLLAR SALES WEIGHTS
    # =====================================================

    def _compute_dollar_sales_weights(
        self,
        meta,
        train_wide
    ):
        """
        Compute M5 dollar-sales weights using the final
        28 days of the training window.
        """

        train_day_cols = list(
            train_wide.columns
        )

        last_28_days = (
            train_day_cols[-28:]
        )

        day_to_week = (
            self.calendar
            .set_index("d")["wm_yr_wk"]
            .to_dict()
        )

        # -------------------------------------------------
        # Convert last 28 days to long format
        # -------------------------------------------------

        long_df = train_wide[
            last_28_days
        ].copy()

        long_df["item_id"] = (
            meta["item_id"].values
        )

        long_df["store_id"] = (
            meta["store_id"].values
        )

        long_df = long_df.melt(
            id_vars=[
                "item_id",
                "store_id",
            ],
            var_name="d",
            value_name="quantity",
        )

        long_df["wm_yr_wk"] = (
            long_df["d"]
            .map(day_to_week)
        )

        # -------------------------------------------------
        # Merge sell price
        # -------------------------------------------------

        long_df = long_df.merge(
            self.sell_prices,
            on=[
                "item_id",
                "store_id",
                "wm_yr_wk",
            ],
            how="left"
        )

        long_df["sell_price"] = (
            long_df["sell_price"]
            .fillna(0)
        )

        long_df["dollar"] = (
            long_df["quantity"]
            * long_df["sell_price"]
        )

        # -------------------------------------------------
        # Bottom-level dollar sales
        # -------------------------------------------------

        dollar_sales = (
            long_df
            .groupby(
                [
                    "item_id",
                    "store_id",
                ]
            )["dollar"]
            .sum()
        )

        bottom_index = (
            pd.MultiIndex.from_frame(
                meta[
                    [
                        "item_id",
                        "store_id",
                    ]
                ]
            )
        )

        dollar_sales = (
            dollar_sales
            .reindex(bottom_index)
            .reset_index(drop=True)
            .fillna(0)
        )

        return dollar_sales

    # =====================================================
    # HIERARCHICAL AGGREGATION
    # =====================================================

    @staticmethod
    def _aggregate_level(
        meta,
        train_wide,
        test_wide,
        forecast_wide,
        dollar_sales,
        keys
    ):
        """
        Aggregate bottom-level series to one of the
        official 12 M5 hierarchy levels.
        """

        # Level 1 = total sales
        if not keys:

            return (
                train_wide
                .sum(axis=0)
                .to_frame()
                .T,

                test_wide
                .sum(axis=0)
                .to_frame()
                .T,

                forecast_wide
                .sum(axis=0)
                .to_frame()
                .T,

                pd.Series([
                    dollar_sales.sum()
                ]),
            )

        group_index = (
            meta
            .groupby(
                keys,
                sort=False
            )
            .ngroup()
        )

        grouped_train = (
            train_wide
            .groupby(group_index)
            .sum()
        )

        grouped_test = (
            test_wide
            .groupby(group_index)
            .sum()
        )

        grouped_forecast = (
            forecast_wide
            .groupby(group_index)
            .sum()
        )

        grouped_weight = (
            dollar_sales
            .groupby(group_index)
            .sum()
        )

        return (
            grouped_train,
            grouped_test,
            grouped_forecast,
            grouped_weight,
        )

    # =====================================================
    # RMSSE
    # =====================================================

    @staticmethod
    def _rmsse_row(
        insample,
        outsample,
        forecast
    ):
        """
        Calculate RMSSE for one time series.
        """

        insample = np.asarray(
            insample,
            dtype=float
        )

        outsample = np.asarray(
            outsample,
            dtype=float
        )

        forecast = np.asarray(
            forecast,
            dtype=float
        )

        # -------------------------------------------------
        # Remove leading zero-sales period
        # -------------------------------------------------

        non_zero = np.nonzero(
            insample > 0
        )[0]

        start = (
            non_zero[0]
            if len(non_zero)
            else 0
        )

        trimmed = insample[
            start:
        ]

        if len(trimmed) < 2:
            return np.nan

        # -------------------------------------------------
        # Naive forecast scaling factor
        # -------------------------------------------------

        scale = np.mean(
            np.diff(trimmed) ** 2
        )

        if (
            scale == 0
            or np.isnan(scale)
        ):
            return np.nan

        # -------------------------------------------------
        # Forecast MSE
        # -------------------------------------------------

        error = np.mean(
            (
                forecast
                - outsample
            ) ** 2
        )

        return np.sqrt(
            error / scale
        )

    # =====================================================
    # WRMSSE
    # =====================================================

    def compute_wrmsse(self):
        """
        Calculate WRMSSE for every M5 hierarchy level
        and return the final 12-level score.
        """

        (
            meta,
            train_wide,
            test_wide,
            forecast_wide,
        ) = self._build_frames()

        dollar_sales = (
            self._compute_dollar_sales_weights(
                meta=meta,
                train_wide=train_wide
            )
        )

        grand_total = (
            dollar_sales.sum()
        )

        if grand_total <= 0:
            raise ValueError(
                "Total dollar-sales weight is zero. "
                "WRMSSE cannot be calculated."
            )

        level_scores = {}

        # -------------------------------------------------
        # Evaluate all 12 hierarchy levels
        # -------------------------------------------------

        for (
            level_name,
            keys
        ) in LEVEL_KEYS.items():

            (
                group_train,
                group_test,
                group_forecast,
                group_weight,
            ) = self._aggregate_level(

                meta=meta,

                train_wide=train_wide,

                test_wide=test_wide,

                forecast_wide=forecast_wide,

                dollar_sales=dollar_sales,

                keys=keys,
            )

            weight_share = (
                group_weight
                / grand_total
            )

            rmsse_values = np.array([
                self._rmsse_row(

                    group_train
                    .iloc[i]
                    .values,

                    group_test
                    .iloc[i]
                    .values,

                    group_forecast
                    .iloc[i]
                    .values,

                )

                for i in range(
                    len(group_train)
                )
            ])

            contributions = (
                rmsse_values
                * weight_share.values
            )

            level_score = float(
                np.nansum(
                    contributions
                )
            )

            level_scores[
                level_name
            ] = level_score

            logger.info(
                f"{level_name}: "
                f"{len(group_train)} groups, "
                f"WRMSSE={level_score:.5f}"
            )

        # -------------------------------------------------
        # Final score
        # -------------------------------------------------

        final_wrmsse = float(
            np.mean(
                list(
                    level_scores.values()
                )
            )
        )

        level_scores[
            "WRMSSE_final"
        ] = final_wrmsse

        logger.info(
            "Final WRMSSE "
            f"(mean across 12 levels): "
            f"{final_wrmsse:.5f}"
        )

        return level_scores

    # =====================================================
    # MLFLOW
    # =====================================================

    def log_into_mlflow(self):
        """
        Load data, calculate WRMSSE, save metrics.json,
        and log the metrics to MLflow/DagsHub.
        """

        # -------------------------------------------------
        # Load data
        # -------------------------------------------------

        self.load_data()

        # -------------------------------------------------
        # Calculate WRMSSE
        # -------------------------------------------------

        scores = (
            self.compute_wrmsse()
        )

        # -------------------------------------------------
        # Save local metrics
        # -------------------------------------------------

        save_json(
            path=Path(
                self.config.metric_file_name
            ),
            data=scores
        )

        # -------------------------------------------------
        # Configure MLflow
        # -------------------------------------------------

        tracking_uri = (
            get_mlflow_tracking_uri()
        )

        mlflow.set_tracking_uri(
            tracking_uri
        )

        # -------------------------------------------------
        # Start MLflow run
        # -------------------------------------------------

        with mlflow.start_run():

            mlflow.log_param(
                "end_train",
                self.config.end_train
            )

            mlflow.log_param(
                "p_horizon",
                self.config.p_horizon
            )

            for (
                metric_name,
                metric_value
            ) in scores.items():

                mlflow.log_metric(
                    metric_name,
                    metric_value
                )

        logger.info(
            "Metrics saved to "
            f"{self.config.metric_file_name} "
            "and logged to MLflow"
        )

        return scores