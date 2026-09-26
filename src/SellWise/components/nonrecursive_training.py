import gc
import pickle
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from SellWise import logger
from SellWise.entity.config_entity import NonRecursiveTrainerConfig


warnings.filterwarnings("ignore")


class NonRecursiveTrainer:
    """
    Trains the three non-recursive SellWise model groups:

    1. Store level
       - 10 models

    2. Store + Category level
       - 30 models

    3. Store + Department level
       - 70 models

    Total:
        110 LightGBM models

    All feature lists, validation windows, model parameters,
    grouping values, and artifact paths are supplied through
    NonRecursiveTrainerConfig.
    """

    def __init__(self, config: NonRecursiveTrainerConfig):
        self.config = config

    @staticmethod
    def _reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Downcast numerical columns to reduce DataFrame memory usage.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        verbose : bool
            Log memory reduction statistics when True.

        Returns
        -------
        pd.DataFrame
            Memory-optimized DataFrame.
        """

        numerics = [
            "int16",
            "int32",
            "int64",
            "float16",
            "float32",
            "float64",
        ]

        start_mem = df.memory_usage().sum() / 1024**2

        for col in df.columns:
            col_type = df[col].dtypes

            if col_type not in numerics:
                continue

            c_min = df[col].min()
            c_max = df[col].max()

            # Skip completely empty numeric columns
            if pd.isna(c_min) or pd.isna(c_max):
                continue

            if str(col_type)[:3] == "int":

                if (
                    c_min > np.iinfo(np.int8).min
                    and c_max < np.iinfo(np.int8).max
                ):
                    df[col] = df[col].astype(np.int8)

                elif (
                    c_min > np.iinfo(np.int16).min
                    and c_max < np.iinfo(np.int16).max
                ):
                    df[col] = df[col].astype(np.int16)

                elif (
                    c_min > np.iinfo(np.int32).min
                    and c_max < np.iinfo(np.int32).max
                ):
                    df[col] = df[col].astype(np.int32)

                else:
                    df[col] = df[col].astype(np.int64)

            else:

                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)

                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)

                else:
                    df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage().sum() / 1024**2

        if verbose:
            reduction = (
                100 * (start_mem - end_mem) / start_mem
                if start_mem > 0
                else 0
            )

            logger.info(
                "Memory usage decreased from %.2f MB to %.2f MB "
                "(%.1f%% reduction)",
                start_mem,
                end_mem,
                reduction,
            )

        return df

    def _load_grid(self) -> pd.DataFrame:
        """
        Load and combine the three preprocessed grid files.
        """

        processed_dir = self.config.processed_data_dir

        logger.info("Loading base grid data")

        grid_1 = pd.read_pickle(
            f"{processed_dir}/grid_part_1.pkl"
        )

        grid_2 = pd.read_pickle(
            f"{processed_dir}/grid_part_2.pkl"
        )[self.config.grid2_colnm]

        grid_3 = pd.read_pickle(
            f"{processed_dir}/grid_part_3.pkl"
        )[self.config.grid3_colnm]

        grid_df = pd.concat(
            [grid_1, grid_2, grid_3],
            axis=1,
        )

        del grid_1, grid_2, grid_3
        gc.collect()

        logger.info(
            "Base grid loaded successfully: %s rows, %s columns",
            grid_df.shape[0],
            grid_df.shape[1],
        )

        return grid_df

    def _prepare_data(
        self,
        base_grid_df: pd.DataFrame,
        level: str,
        store: str,
        cat: str = None,
        dept: str = None,
    ) -> pd.DataFrame:
        """
        Prepare training data for one store/category/department group.
        """

        processed_dir = self.config.processed_data_dir

        grid_df = base_grid_df[
            base_grid_df["store_id"] == store
        ].copy()

        if level == "store_cat":

            if cat is None:
                raise ValueError(
                    "cat must be provided when level='store_cat'"
                )

            grid_df = grid_df[
                grid_df["cat_id"] == cat
            ].copy()

        elif level == "store_dept":

            if dept is None:
                raise ValueError(
                    "dept must be provided when level='store_dept'"
                )

            grid_df = grid_df[
                grid_df["dept_id"] == dept
            ].copy()

        elif level != "store":
            raise ValueError(
                f"Unknown training level: {level}"
            )

        # Restrict training history
        grid_df = grid_df[
            grid_df["d"] >= self.config.first_day
        ].copy()

        if grid_df.empty:
            raise ValueError(
                f"No rows found for "
                f"level={level}, store={store}, "
                f"cat={cat}, dept={dept}"
            )

        # -------------------------------------------------
        # Lag features
        # -------------------------------------------------

        lag_df = pd.read_pickle(
            f"{processed_dir}/lags_df_28.pkl"
        )[self.config.lag_colnm]

        try:
            lag_df = lag_df.loc[grid_df.index]
        except KeyError as exc:
            raise KeyError(
                "Lag feature indices do not align with grid indices"
            ) from exc

        grid_df = pd.concat(
            [grid_df, lag_df],
            axis=1,
        )

        del lag_df
        gc.collect()

        # -------------------------------------------------
        # Mean encoding features
        # -------------------------------------------------

        mean_enc_cols = (
            self.config.mean_enc_by_level[level]
        )

        mean_enc_df = pd.read_pickle(
            f"{processed_dir}/mean_encoding_df.pkl"
        )[mean_enc_cols]

        try:
            mean_enc_df = mean_enc_df.loc[
                grid_df.index
            ]
        except KeyError as exc:
            raise KeyError(
                "Mean encoding indices do not align "
                "with grid indices"
            ) from exc

        grid_df = pd.concat(
            [grid_df, mean_enc_df],
            axis=1,
        )

        del mean_enc_df
        gc.collect()

        # -------------------------------------------------
        # Memory optimization
        # -------------------------------------------------

        grid_df = self._reduce_mem_usage(
            grid_df,
            verbose=False,
        )

        return grid_df

    def _train_one_group(
        self,
        base_grid_df: pd.DataFrame,
        level: str,
        store: str,
        cat: str = None,
        dept: str = None,
    ) -> None:
        """
        Train one LightGBM model for a given hierarchy group.
        """

        remove_feature = (
            self.config.remove_feature_by_level[level]
        )

        cv = self.config.cv

        if cv not in self.config.validation:
            raise ValueError(
                f"Unknown validation split '{cv}'. "
                f"Available splits: "
                f"{list(self.config.validation.keys())}"
            )

        grid_df = self._prepare_data(
            base_grid_df=base_grid_df,
            level=level,
            store=store,
            cat=cat,
            dept=dept,
        )

        # -------------------------------------------------
        # Model features
        # -------------------------------------------------

        model_var = grid_df.columns[
            ~grid_df.columns.isin(remove_feature)
        ].tolist()

        if not model_var:
            raise ValueError(
                f"No model features available for {level}"
            )

        # -------------------------------------------------
        # Train / validation split
        # -------------------------------------------------

        day_range = self.config.validation[cv]

        train_end = day_range[0]
        validation_end = day_range[1]

        tr_mask = (
            (grid_df["d"] >= self.config.first_day)
            & (grid_df["d"] <= train_end)
        )

        vl_mask = (
            (grid_df["d"] > train_end)
            & (grid_df["d"] <= validation_end)
        )

        if tr_mask.sum() == 0:
            raise ValueError(
                f"No training rows found for "
                f"{level}/{store}/{cat}/{dept}"
            )

        if vl_mask.sum() == 0:
            raise ValueError(
                f"No validation rows found for "
                f"{level}/{store}/{cat}/{dept}"
            )

        validation_target = grid_df.loc[
            vl_mask,
            "sales",
        ]

        if validation_target.isna().any():
            raise ValueError(
                f"Validation split '{cv}' contains "
                f"NaN targets for "
                f"{level}/{store}/{cat}/{dept}. "
                f"Use a validation range with known sales."
            )

        # -------------------------------------------------
        # LightGBM datasets
        # -------------------------------------------------

        train_data = lgb.Dataset(
            grid_df.loc[
                tr_mask,
                model_var,
            ],
            label=grid_df.loc[
                tr_mask,
                "sales",
            ],
            free_raw_data=False,
        )

        valid_data = lgb.Dataset(
            grid_df.loc[
                vl_mask,
                model_var,
            ],
            label=validation_target,
            reference=train_data,
            free_raw_data=False,
        )

        logger.info(
            "Training model | "
            "level=%s | store=%s | cat=%s | dept=%s | "
            "train_rows=%s | valid_rows=%s",
            level,
            store,
            cat,
            dept,
            tr_mask.sum(),
            vl_mask.sum(),
        )

        # -------------------------------------------------
        # Train model
        # -------------------------------------------------

        model = lgb.train(
            params=self.config.lgb_params,
            train_set=train_data,
            valid_sets=[valid_data],
            valid_names=["valid"],
            callbacks=[
                lgb.log_evaluation(period=100),
                lgb.early_stopping(
                    stopping_rounds=(
                        self.config.early_stopping_rounds
                    )
                ),
            ],
        )

        logger.info(
            "Training complete | "
            "best_iteration=%s",
            model.best_iteration,
        )

        # -------------------------------------------------
        # Model artifact name
        # -------------------------------------------------

        if level == "store":

            model_name = (
                f"{self.config.models_dir}/"
                f"non_recur_model_{store}.bin"
            )

        elif level == "store_cat":

            model_name = (
                f"{self.config.models_dir}/"
                f"non_recur_model_{store}_{cat}.bin"
            )

        elif level == "store_dept":

            model_name = (
                f"{self.config.models_dir}/"
                f"non_recur_model_{store}_{dept}.bin"
            )

        else:
            raise ValueError(
                f"Unknown level: {level}"
            )

        # -------------------------------------------------
        # Save model
        # -------------------------------------------------

        with open(model_name, "wb") as file:
            pickle.dump(
                model,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        logger.info(
            "Saved model: %s",
            model_name,
        )

        # -------------------------------------------------
        # Cleanup
        # -------------------------------------------------

        del (
            grid_df,
            train_data,
            valid_data,
            validation_target,
            model,
            tr_mask,
            vl_mask,
        )

        gc.collect()

    def train_store_level(self) -> None:
        """
        Train one non-recursive model per store.

        Total models: 10
        """

        logger.info(
            "Starting non-recursive STORE-level training"
        )

        base_grid_df = self._load_grid()

        for store in self.config.stores:

            logger.info(
                "Training store: %s",
                store,
            )

            self._train_one_group(
                base_grid_df=base_grid_df,
                level="store",
                store=store,
            )

        del base_grid_df
        gc.collect()

        logger.info(
            "STORE-level training complete"
        )

    def train_store_cat_level(self) -> None:
        """
        Train one non-recursive model per store/category.

        Total models: 30
        """

        logger.info(
            "Starting non-recursive "
            "STORE+CATEGORY-level training"
        )

        base_grid_df = self._load_grid()

        for store in self.config.stores:

            for cat in self.config.cats:

                logger.info(
                    "Training store/category: %s / %s",
                    store,
                    cat,
                )

                self._train_one_group(
                    base_grid_df=base_grid_df,
                    level="store_cat",
                    store=store,
                    cat=cat,
                )

        del base_grid_df
        gc.collect()

        logger.info(
            "STORE+CATEGORY-level training complete"
        )

    def train_store_dept_level(self) -> None:
        """
        Train one non-recursive model per store/department.

        Total models: 70
        """

        logger.info(
            "Starting non-recursive "
            "STORE+DEPARTMENT-level training"
        )

        base_grid_df = self._load_grid()

        for store in self.config.stores:

            for dept in self.config.depts:

                logger.info(
                    "Training store/department: %s / %s",
                    store,
                    dept,
                )

                self._train_one_group(
                    base_grid_df=base_grid_df,
                    level="store_dept",
                    store=store,
                    dept=dept,
                )

        del base_grid_df
        gc.collect()

        logger.info(
            "STORE+DEPARTMENT-level training complete"
        )

    def run(self) -> None:
        """
        Execute all three non-recursive training stages.

        Models generated:
            Store              : 10
            Store + Category   : 30
            Store + Department : 70

            Total              : 110
        """

        logger.info(
            "=" * 60
        )

        logger.info(
            "Starting SellWise Non-Recursive Training Pipeline"
        )

        logger.info(
            "=" * 60
        )

        self.train_store_level()

        self.train_store_cat_level()

        self.train_store_dept_level()

        logger.info(
            "=" * 60
        )

        logger.info(
            "Non-recursive training complete. "
            "110 models saved successfully."
        )

        logger.info(
            "=" * 60
        )