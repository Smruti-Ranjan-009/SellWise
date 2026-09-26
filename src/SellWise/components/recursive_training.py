import gc
import pickle
import numpy as np, pandas as pd
import lightgbm as lgb
import warnings
from SellWise import logger
from SellWise.entity.config_entity import RecursiveTrainerConfig

warnings.filterwarnings('ignore')

class RecursiveTrainer:
    """
    Unifies A1's 3 recursive training notebooks (1-1 store, 1-2 store+cat,
    1-3 store+dept). Retains positional column slicing (.iloc[:, N:]) and 
    tmp rolling feature inclusion exactly matching A1's recursive approach.

    Includes early stopping by splitting train and validation sets without 
    data overlap.
    """

    def __init__(self, config: RecursiveTrainerConfig):
        self.config = config

    def _load_grid(self):
        p = self.config.processed_data_dir
        grid_1 = pd.read_pickle(f"{p}/grid_part_1.pkl")
        grid_2 = pd.read_pickle(f"{p}/grid_part_2.pkl").iloc[:, 2:]   # skip id, d
        grid_3 = pd.read_pickle(f"{p}/grid_part_3.pkl").iloc[:, 2:]   # skip id, d
        grid_df = pd.concat([grid_1, grid_2, grid_3], axis=1)
        del grid_1, grid_2, grid_3
        gc.collect()
        return grid_df

    def _get_data_by_group(self, base_grid_df, level, store, cat=None, dept=None):
        p = self.config.processed_data_dir
        start_train = self.config.start_train_by_level[level]

        grid_df = base_grid_df[base_grid_df['d'] >= start_train]
        grid_df = grid_df[grid_df['store_id'] == store]
        if level == 'store_cat':
            grid_df = grid_df[grid_df['cat_id'] == cat]
        elif level == 'store_dept':
            grid_df = grid_df[grid_df['dept_id'] == dept]

        mean_enc_cols = self.config.mean_enc_by_level[level]
        mean_enc = pd.read_pickle(f"{p}/mean_encoding_df.pkl")[mean_enc_cols]
        mean_enc = mean_enc[mean_enc.index.isin(grid_df.index)]

        lag = pd.read_pickle(f"{p}/lags_df_28.pkl").iloc[:, 3:]  # skip id, d, sales -- includes tmp cols
        lag = lag[lag.index.isin(grid_df.index)]

        grid_df = pd.concat([grid_df, mean_enc], axis=1)
        del mean_enc
        grid_df = pd.concat([grid_df, lag], axis=1)
        del lag
        gc.collect()

        remove_feature = self.config.remove_feature_by_level[level]
        features = [col for col in list(grid_df) if col not in remove_feature]
        grid_df = grid_df[['id', 'd', 'sales'] + features]
        grid_df = grid_df.reset_index(drop=True)

        return grid_df, features

    def _train_one_group(self, base_grid_df, level, store, cat=None, dept=None):
        end_train = self.config.end_train
        p_horizon = self.config.p_horizon

        grid_df, features = self._get_data_by_group(base_grid_df, level, store, cat=cat, dept=dept)

        # ADJUSTED FOR EARLY STOPPING:
        # Validation set: last p_horizon days up to end_train
        valid_mask = (grid_df['d'] <= end_train) & (grid_df['d'] > (end_train - p_horizon))
        # Training set: history up to start of validation window (prevents data leakage)
        train_mask = grid_df['d'] <= (end_train - p_horizon)

        preds_mask = (grid_df['d'] > (end_train - 100)) & (grid_df['d'] <= end_train + p_horizon)

        train_data = lgb.Dataset(grid_df[train_mask][features], label=grid_df[train_mask]['sales'])
        valid_data = lgb.Dataset(grid_df[valid_mask][features], label=grid_df[valid_mask]['sales'])

        # A1's Exact Test Skeleton for Recursive Prediction
        test_df = grid_df[preds_mask].reset_index(drop=True)
        keep_cols = [col for col in list(test_df) if '_tmp_' not in col]
        test_df = test_df[keep_cols]

        d_sales = test_df[['d', 'sales']]
        substitute = d_sales['sales'].values.copy()
        substitute[(d_sales['d'] > end_train)] = np.nan
        test_df['sales'] = substitute

        if level == 'store':
            test_name = f"{self.config.processed_data_dir}/test_{store}.pkl"
            model_name = f"{self.config.models_dir}/lgb_model_{store}_v{self.config.ver}.bin"
            lgb_params = dict(self.config.lgb_params_store)
        elif level == 'store_cat':
            test_name = f"{self.config.processed_data_dir}/test_{store}_{cat}.pkl"
            model_name = f"{self.config.models_dir}/lgb_model_{store}_{cat}_v{self.config.ver}.bin"
            lgb_params = dict(self.config.lgb_params_default)
        else:
            test_name = f"{self.config.processed_data_dir}/test_{store}_{dept}.pkl"
            model_name = f"{self.config.models_dir}/lgb_model_{store}_{dept}_v{self.config.ver}.bin"
            lgb_params = dict(self.config.lgb_params_default)

        test_df.to_pickle(test_name)
        del test_df, d_sales, substitute

        # LightGBM Training with Early Stopping
        estimator = lgb.train(
            lgb_params, 
            train_data,
            valid_sets=[valid_data],
            valid_names=['valid'],
            callbacks=[
                lgb.log_evaluation(period=100),
                lgb.early_stopping(stopping_rounds=self.config.early_stopping_rounds),
            ]
        )

        pickle.dump(estimator, open(model_name, 'wb'))
        logger.info(f"Saved {model_name}")

        del grid_df, train_data, valid_data, estimator, train_mask, valid_mask, preds_mask
        gc.collect()

    def train_store_level(self):
        """A1's 1-1: one model per store (10 models)."""
        logger.info("Training recursive STORE-level models")
        base_grid_df = self._load_grid()
        for store in self.config.stores:
            logger.info(f"  {store}")
            self._train_one_group(base_grid_df, 'store', store)
        del base_grid_df
        gc.collect()

    def train_store_cat_level(self):
        """A1's 1-2: one model per store x category (30 models)."""
        logger.info("Training recursive STORE+CATEGORY-level models")
        base_grid_df = self._load_grid()
        for store in self.config.stores:
            for cat in self.config.cats:
                logger.info(f"  {store} / {cat}")
                self._train_one_group(base_grid_df, 'store_cat', store, cat=cat)
        del base_grid_df
        gc.collect()

    def train_store_dept_level(self):
        """A1's 1-3: one model per store x department (70 models)."""
        logger.info("Training recursive STORE+DEPARTMENT-level models")
        base_grid_df = self._load_grid()
        for store in self.config.stores:
            for dept in self.config.depts:
                logger.info(f"  {store} / {dept}")
                self._train_one_group(base_grid_df, 'store_dept', store, dept=dept)
        del base_grid_df
        gc.collect()

    def run(self):
        """Runs all 3 levels in sequence: 10 + 30 + 70 = 110 recursive models total."""
        self.train_store_level()
        self.train_store_cat_level()
        self.train_store_dept_level()
        logger.info("All recursive training complete: 110 models saved")