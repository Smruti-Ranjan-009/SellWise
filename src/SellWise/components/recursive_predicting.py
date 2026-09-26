import gc
import time
import pickle
import numpy as np, pandas as pd
import warnings
from SellWise import logger
from SellWise.entity.config_entity import RecursivePredictorConfig

warnings.filterwarnings('ignore')

class RecursivePredictor:
    """
    Unifies A1's 3 recursive predict scripts (1-1 store, 1-2 store+cat,
    1-3 store+dept). Reads the test_{group}.pkl skeletons the recursive
    trainer saved, then predicts day-by-day: each day's predictions get
    written back into the running "sales" series, and the next day's
    rolling_mean_tmp_* features are recomputed from that updated series
    before the next day is predicted.

    A1's original make_lag() (plain single-day shifted lag) is defined in
    their code but never actually called in the predict loop -- only
    make_lag_roll() (rolling means of shifted values) is used, since the
    model's static lag_28..42 features don't need recomputation. Reproduced
    here accordingly: only the rolling recomputation is implemented.

    Uses .loc for the recursive write-back instead of A1's chained
    indexing (base_test[TARGET][mask] = ...), which is unreliable on
    modern pandas.
    """

    def __init__(self, config: RecursivePredictorConfig):
        self.config = config

    def _get_data_by_group(self, level, store, cat=None, dept=None):
        """Same feature-construction logic as RecursiveTrainer -- used here
        only to derive the canonical MODEL_FEATURES list per level."""
        p = self.config.processed_data_dir
        start_train = 0 if level == 'store' else (700 if level == 'store_cat' else 710)

        grid_1 = pd.read_pickle(f"{p}/grid_part_1.pkl")
        grid_2 = pd.read_pickle(f"{p}/grid_part_2.pkl").iloc[:, 2:]
        grid_3 = pd.read_pickle(f"{p}/grid_part_3.pkl").iloc[:, 2:]
        df = pd.concat([grid_1, grid_2, grid_3], axis=1)
        del grid_1, grid_2, grid_3
        gc.collect()

        df = df[df['d'] >= start_train]
        df = df[df['store_id'] == store]
        if level == 'store_cat':
            df = df[df['cat_id'] == cat]
        elif level == 'store_dept':
            df = df[df['dept_id'] == dept]

        mean_enc_cols = self.config.mean_enc_by_level[level]
        mean_enc = pd.read_pickle(f"{p}/mean_encoding_df.pkl")[mean_enc_cols]
        mean_enc = mean_enc[mean_enc.index.isin(df.index)]

        lag = pd.read_pickle(f"{p}/lags_df_28.pkl").iloc[:, 3:]
        lag = lag[lag.index.isin(df.index)]

        df = pd.concat([df, mean_enc], axis=1)
        df = pd.concat([df, lag], axis=1)
        del mean_enc, lag
        gc.collect()

        remove_feature = self.config.remove_feature_by_level[level]
        features = [col for col in list(df) if col not in remove_feature]
        return features

    def _load_base_test(self, level):
        """Reassembles the full test skeleton for a level from the
        per-group test_{group}.pkl files, re-adding the grouping id
        column(s) that were dropped as features during training."""
        p = self.config.processed_data_dir
        base_test = pd.DataFrame()

        if level == 'store':
            for store in self.config.stores:
                temp_df = pd.read_pickle(f"{p}/test_{store}.pkl")
                temp_df['store_id'] = store
                base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
        elif level == 'store_cat':
            for store in self.config.stores:
                for cat in self.config.cats:
                    temp_df = pd.read_pickle(f"{p}/test_{store}_{cat}.pkl")
                    temp_df['store_id'] = store
                    temp_df['cat_id'] = cat
                    base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)
        else:
            for store in self.config.stores:
                for dept in self.config.depts:
                    temp_df = pd.read_pickle(f"{p}/test_{store}_{dept}.pkl")
                    temp_df['store_id'] = store
                    temp_df['dept_id'] = dept
                    base_test = pd.concat([base_test, temp_df]).reset_index(drop=True)

        return base_test

    def _make_lag_roll(self, base_test, shift_day, roll_wind):
        col_name = f"rolling_mean_tmp_{shift_day}_{roll_wind}"
        lag_df = base_test[['id', 'd', 'sales']].copy()
        lag_df[col_name] = lag_df.groupby(['id'])['sales'].transform(
            lambda x: x.shift(shift_day).rolling(roll_wind).mean()
        )
        return lag_df[[col_name]]

    def _model_path(self, level, store, cat=None, dept=None):
        m = self.config.models_dir
        v = self.config.ver
        if level == 'store':
            return f"{m}/lgb_model_{store}_v{v}.bin"
        elif level == 'store_cat':
            return f"{m}/lgb_model_{store}_{cat}_v{v}.bin"
        else:
            return f"{m}/lgb_model_{store}_{dept}_v{v}.bin"

    def predict_level(self, level):
        end_train = self.config.end_train
        p_horizon = self.config.p_horizon

        if level == 'store':
            last_features = self._get_data_by_group(level, self.config.stores[-1])
            groups = [(s, None, None) for s in self.config.stores]
        elif level == 'store_cat':
            last_features = self._get_data_by_group(level, self.config.stores[-1], cat=self.config.cats[-1])
            groups = [(s, c, None) for s in self.config.stores for c in self.config.cats]
        else:
            last_features = self._get_data_by_group(level, self.config.stores[-1], dept=self.config.depts[-1])
            groups = [(s, None, d) for s in self.config.stores for d in self.config.depts]

        model_features = last_features

        base_test = self._load_base_test(level)
        all_preds = pd.DataFrame()

        logger.info(f"Predicting recursive {level}-level: {len(groups)} groups, {p_horizon} days")
        main_time = time.time()

        for predict_day in range(1, p_horizon + 1):
            start_time = time.time()

            grid_df = base_test.copy()
            roll_frames = [
                self._make_lag_roll(base_test, shift, window)
                for shift, window in self.config.rols_split
            ]
            grid_df = pd.concat([grid_df] + roll_frames, axis=1)
            del roll_frames
            gc.collect()

            day_mask = base_test['d'] == (end_train + predict_day)

            for store, cat, dept in groups:
                model_path = self._model_path(level, store, cat=cat, dept=dept)
                estimator = pickle.load(open(model_path, 'rb'))

                group_mask = base_test['store_id'] == store
                if level == 'store_cat':
                    group_mask = group_mask & (base_test['cat_id'] == cat)
                elif level == 'store_dept':
                    group_mask = group_mask & (base_test['dept_id'] == dept)

                mask = day_mask & group_mask
                base_test.loc[mask, 'sales'] = estimator.predict(grid_df.loc[mask, model_features])

            temp_df = base_test.loc[day_mask, ['id', 'sales']].copy()
            temp_df.columns = ['id', f'F{predict_day}']

            if 'id' in list(all_preds):
                all_preds = all_preds.merge(temp_df, on=['id'], how='left')
            else:
                all_preds = temp_df.copy()

            logger.info(
                f"  day {predict_day}/{p_horizon}: "
                f"{(time.time() - start_time):.1f}s, "
                f"total {(time.time() - main_time) / 60:.2f} min, "
                f"predicted sum {temp_df[f'F{predict_day}'].sum():.1f}"
            )
            del temp_df

        return all_preds.reset_index(drop=True)

    def export_predictions(self, all_preds, level):
        submission_ids = pd.read_csv(self.config.sample_submission_path)[['id']]
        submission = submission_ids.merge(all_preds, on=['id'], how='left').fillna(0)
        out_path = f"{self.config.submission_dir}/submission_recursive_{level}.csv"
        submission.to_csv(out_path, index=False)
        logger.info(f"Saved {out_path}")
        return out_path

    def run(self):
        """Predicts and exports all 3 recursive model families."""
        for level in ['store', 'store_cat', 'store_dept']:
            preds = self.predict_level(level)
            self.export_predictions(preds, level)
        logger.info("All recursive prediction complete: 3 submission files saved")
