import gc
import time
import pickle
import numpy as np, pandas as pd
import warnings
from SellWise import logger
from SellWise.entity.config_entity import NonRecursivePredictorConfig

warnings.filterwarnings('ignore')

class NonRecursivePredictor:
    """
    Unifies A1's 3 non-recursive predict scripts (2-1 store, 2-2 store+cat,
    2-3 store+dept). For every group, rebuilds the feature grid, loads the
    pre-trained model, predicts the 28-day horizon in one shot (all lag /
    rolling features are static, so no day-by-day write-back is needed),
    and pivots the result into an id x d table saved as a per-group CSV.
    All per-group CSVs of a level are then summed onto the evaluation half
    of sample_submission (ids are disjoint across groups, so the sum is
    effectively a union), then expanded to all sample_submission ids and exported
    in the same format as the recursive predictor's files.

    Per-level differences (same as the original 3 notebooks):
      - store       : filter store_id;                 mean_enc = store_dept + item_state
      - store_cat   : filter store_id + cat_id;        mean_enc = store_dept + item_store
      - store_dept  : filter store_id + dept_id;       mean_enc = item_store
    Only the level-specific remove_feature / mean_enc lists come from params.
    """

    # file-name tag used by A1 for each level's per-group CSVs
    LEVEL_TAG = {
        'store': 'onlystore',
        'store_cat': 'storeandcat',
        'store_dept': 'storeanddept',
    }
    # A1's model file naming and the evaluation-half offset of sample_submission
    MODEL_PREFIX = 'non_recur_model_'
    EVAL_ROW_OFFSET = 30490

    def __init__(self, config: NonRecursivePredictorConfig):
        self.config = config

    @staticmethod
    def _reduce_mem_usage(df, verbose=False):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose:
            logger.info('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
                end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df

    def _prepare_data(self, level, store, cat=None, dept=None):
        """Same feature grid A1's prepare_data() builds: grid_1 + selected
        grid_2/grid_3 columns, filtered to the group and d >= first_day,
        then selected lag columns and level-specific mean-encoding columns."""
        p = self.config.processed_data_dir

        grid_1 = pd.read_pickle(f"{p}/grid_part_1.pkl")
        grid_2 = pd.read_pickle(f"{p}/grid_part_2.pkl")[self.config.grid2_colnm]
        grid_3 = pd.read_pickle(f"{p}/grid_part_3.pkl")[self.config.grid3_colnm]

        grid_df = pd.concat([grid_1, grid_2, grid_3], axis=1)
        del grid_1, grid_2, grid_3
        gc.collect()

        if level == 'store':
            grid_df = grid_df[grid_df['store_id'] == store]
        elif level == 'store_cat':
            grid_df = grid_df[(grid_df['store_id'] == store) & (grid_df['cat_id'] == cat)]
        else:
            grid_df = grid_df[(grid_df['store_id'] == store) & (grid_df['dept_id'] == dept)]
        grid_df = grid_df[grid_df['d'] >= self.config.first_day]

        lag = pd.read_pickle(f"{p}/lags_df_28.pkl")[self.config.lag_colnm]
        lag = lag[lag.index.isin(grid_df.index)]

        grid_df = pd.concat([grid_df, lag], axis=1)
        del lag
        gc.collect()

        mean_enc = pd.read_pickle(f"{p}/mean_encoding_df.pkl")[self.config.mean_enc_by_level[level]]
        mean_enc = mean_enc[mean_enc.index.isin(grid_df.index)]

        grid_df = pd.concat([grid_df, mean_enc], axis=1)
        del mean_enc
        gc.collect()

        return self._reduce_mem_usage(grid_df)

    def _model_path(self, level, store, cat=None, dept=None):
        m = self.config.models_dir
        pre = self.MODEL_PREFIX
        if level == 'store':
            return f"{m}/{pre}{store}.bin"
        elif level == 'store_cat':
            return f"{m}/{pre}{store}_{cat}.bin"
        else:
            return f"{m}/{pre}{store}_{dept}.bin"

    def predict_level(self, level):
        """Predicts every group of a level; returns the per-group CSV paths."""
        end_train = self.config.end_train
        p_horizon = self.config.p_horizon
        remove_feature = self.config.remove_feature_by_level[level]

        if level == 'store':
            groups = [(s, None, None) for s in self.config.stores]
        elif level == 'store_cat':
            groups = [(s, c, None) for s in self.config.stores for c in self.config.cats]
        else:
            groups = [(s, None, d) for s in self.config.stores for d in self.config.depts]

        logger.info(f"Predicting non-recursive {level}-level: {len(groups)} groups, {p_horizon} days")
        main_time = time.time()
        files = []

        for store, cat, dept in groups:
            start_time = time.time()
            group_name = "_".join([g for g in (store, cat, dept) if g])

            grid_df = self._prepare_data(level, store, cat=cat, dept=dept)
            model_var = grid_df.columns[~grid_df.columns.isin(remove_feature)]

            vl_mask = (grid_df['d'] > end_train) & (grid_df['d'] <= end_train + p_horizon)

            estimator = pickle.load(open(self._model_path(level, store, cat=cat, dept=dept), 'rb'))

            indice = grid_df[vl_mask].index.tolist()
            prediction = pd.DataFrame({'y_pred': estimator.predict(grid_df[vl_mask][model_var])})
            prediction.index = indice

            out_path = f"{self.config.log_dir}/submission_{self.LEVEL_TAG[level]}_{group_name}_{self.config.cv}.csv"
            pd.concat([grid_df.loc[indice, ['id', 'd']], prediction], axis=1) \
                .pivot(index='id', columns='d', values='y_pred') \
                .reset_index() \
                .set_index('id') \
                .to_csv(out_path)
            files.append(out_path)

            del grid_df, estimator, vl_mask, prediction
            gc.collect()

            logger.info(
                f"  {group_name}: {(time.time() - start_time):.1f}s, "
                f"total {(time.time() - main_time) / 60:.2f} min"
            )

        return files

    def export_predictions(self, files, level):
        """Sums the per-group CSVs onto the evaluation rows of sample_submission."""
        submission = pd.read_csv(self.config.sample_submission_path).set_index('id') \
            .iloc[self.EVAL_ROW_OFFSET:]
        sub_id = pd.DataFrame({'id': submission.index.tolist()})

        fcol = [f'F{i}' for i in range(1, self.config.p_horizon + 1)]

        sub_copy = submission.copy()
        for file in files:
            temp = pd.read_csv(file)
            temp.columns = ['id'] + fcol
            sub_copy += sub_id.merge(temp, how='left', on='id').set_index('id').fillna(0)
        sub_copy.columns = fcol

        # expand to all sample_submission ids (validation half = 0), same
        # shape as RecursivePredictor.export_predictions, so both families
        # can be ensembled with a plain merge on 'id'
        submission_ids = pd.read_csv(self.config.sample_submission_path)[['id']]
        full = submission_ids.merge(sub_copy.reset_index(), on='id', how='left').fillna(0)

        out_path = f"{self.config.submission_dir}/submission_nonrecursive_{level}.csv"
        full.to_csv(out_path, index=False)
        logger.info(f"Saved {out_path}")
        return out_path

    def run(self):
        """Predicts and exports all 3 non-recursive model families."""
        for level in ['store', 'store_cat', 'store_dept']:
            files = self.predict_level(level)
            self.export_predictions(files, level)
        logger.info("All non-recursive prediction complete: 3 submission files saved")