import os
import time
import gc
import warnings
from math import ceil
import numpy as np
import pandas as pd
from SellWise import logger
from SellWise.entity.config_entity import DataPreprocessingConfig

warnings.filterwarnings('ignore')

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.TARGET = 'sales'
        self.END_TRAIN = 1941
        self.MAIN_INDEX = ['id', 'd']

    @staticmethod
    def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
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
            logger.info(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
        return df

    @staticmethod
    def merge_by_concat(df1: pd.DataFrame, df2: pd.DataFrame, merge_on: list) -> pd.DataFrame:
        merged_gf = df1[merge_on].merge(df2, on=merge_on, how='left')
        new_columns = [col for col in list(merged_gf) if col not in merge_on]
        return pd.concat([df1, merged_gf[new_columns]], axis=1)

    def create_grid_part_1(self):
        logger.info("Starting Grid & Part 1 processing...")
        train_df = pd.read_csv(self.config.sales_path)
        prices_df = pd.read_csv(self.config.sell_prices_path)
        calendar_df = pd.read_csv(self.config.calendar_path)

        if 'id' not in train_df.columns:
            train_df.insert(0, 'id', train_df['item_id'] + '_' + train_df['store_id'] + '_evaluation')

        index_columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        grid_df = pd.melt(train_df, id_vars=index_columns, var_name='d', value_name=self.TARGET)

        add_grid = pd.DataFrame()
        for i in range(1, 29):
            temp_df = train_df[index_columns].drop_duplicates()
            temp_df['d'] = f'd_{self.END_TRAIN + i}'
            temp_df[self.TARGET] = np.nan
            add_grid = pd.concat([add_grid, temp_df])

        grid_df = pd.concat([grid_df, add_grid]).reset_index(drop=True)
        del temp_df, add_grid, train_df

        for col in index_columns:
            grid_df[col] = grid_df[col].astype('category')

        release_df = prices_df.groupby(['store_id', 'item_id'])['wm_yr_wk'].agg(['min']).reset_index()
        release_df.columns = ['store_id', 'item_id', 'release']

        grid_df = self.merge_by_concat(grid_df, release_df, ['store_id', 'item_id'])
        del release_df

        grid_df = self.merge_by_concat(grid_df, calendar_df[['wm_yr_wk', 'd']], ['d'])
        grid_df = grid_df[grid_df['wm_yr_wk'] >= grid_df['release']].reset_index(drop=True)

        grid_df['release'] = (grid_df['release'] - grid_df['release'].min()).astype(np.int16)

        grid_df.to_pickle(self.config.grid_part_1)
        logger.info(f"Saved grid_part_1 to {self.config.grid_part_1} with shape {grid_df.shape}")

    def create_grid_part_2(self):
        logger.info("Starting Part 2 (Prices) processing...")
        prices_df = pd.read_csv(self.config.sell_prices_path)
        calendar_df = pd.read_csv(self.config.calendar_path)
        grid_df = pd.read_pickle(self.config.grid_part_1)

        prices_df['price_max'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
        prices_df['price_min'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('min')
        prices_df['price_std'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('std')
        prices_df['price_mean'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')
        prices_df['price_norm'] = prices_df['sell_price'] / prices_df['price_max']
        prices_df['price_nunique'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('nunique')
        prices_df['item_nunique'] = prices_df.groupby(['store_id', 'sell_price'])['item_id'].transform('nunique')

        calendar_prices = calendar_df[['wm_yr_wk', 'month', 'year']].drop_duplicates(subset=['wm_yr_wk'])
        prices_df = prices_df.merge(calendar_prices[['wm_yr_wk', 'month', 'year']], on=['wm_yr_wk'], how='left')

        prices_df['price_momentum'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform(lambda x: x.shift(1))
        prices_df['price_momentum_m'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id', 'month'])['sell_price'].transform('mean')
        prices_df['price_momentum_y'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id', 'year'])['sell_price'].transform('mean')

        del prices_df['month'], prices_df['year']

        grid_df = self.reduce_mem_usage(grid_df)
        prices_df = self.reduce_mem_usage(prices_df)

        original_columns = list(grid_df)
        grid_df = grid_df.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        keep_columns = [col for col in list(grid_df) if col not in original_columns]
        grid_df = grid_df[self.MAIN_INDEX + keep_columns]
        grid_df = self.reduce_mem_usage(grid_df)

        grid_df.to_pickle(self.config.grid_part_2)
        logger.info(f"Saved grid_part_2 to {self.config.grid_part_2} with shape {grid_df.shape}")

    def create_grid_part_3(self):
        logger.info("Starting Part 3 (Calendar/Dates) processing...")
        calendar_df = pd.read_csv(self.config.calendar_path)
        grid_df = pd.read_pickle(self.config.grid_part_1)
        grid_df = grid_df[self.MAIN_INDEX]

        icols = ['date', 'd', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI']
        grid_df = grid_df.merge(calendar_df[icols], on=['d'], how='left')

        cat_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI']
        for col in cat_cols:
            grid_df[col] = grid_df[col].astype('category')

        grid_df['date'] = pd.to_datetime(grid_df['date'])
        grid_df['tm_d'] = grid_df['date'].dt.day.astype(np.int8)
        grid_df['tm_w'] = grid_df['date'].dt.isocalendar().week.astype(np.int8)
        grid_df['tm_m'] = grid_df['date'].dt.month.astype(np.int8)
        grid_df['tm_y'] = grid_df['date'].dt.year
        grid_df['tm_y'] = (grid_df['tm_y'] - grid_df['tm_y'].min()).astype(np.int8)
        grid_df['tm_wm'] = grid_df['tm_d'].apply(lambda x: ceil(x / 7)).astype(np.int8)
        grid_df['tm_dw'] = grid_df['date'].dt.dayofweek.astype(np.int8)
        grid_df['tm_w_end'] = (grid_df['tm_dw'] >= 5).astype(np.int8)

        del grid_df['date']
        grid_df.to_pickle(self.config.grid_part_3)
        logger.info(f"Saved grid_part_3 to {self.config.grid_part_3} with shape {grid_df.shape}")

    def finalize_part_1(self):
        logger.info("Finalizing grid_part_1 formatting...")
        grid_df = pd.read_pickle(self.config.grid_part_1)
        grid_df['d'] = grid_df['d'].apply(lambda x: x[2:]).astype(np.int16)
        del grid_df['wm_yr_wk']
        grid_df.to_pickle(self.config.grid_part_1)

    def create_lags(self, shift_day: int = 28):
        logger.info(f"Creating lag features (shift={shift_day})...")
        grid_df = pd.read_pickle(self.config.grid_part_1)[['id', 'd', self.TARGET]]

        lag_days = [col for col in range(shift_day, shift_day + 15)]
        grid_df = grid_df.assign(**{
            f'{self.TARGET}_lag_{l}': grid_df.groupby(['id'])[self.TARGET].transform(lambda x: x.shift(l))
            for l in lag_days
        })

        for col in list(grid_df):
            if 'lag' in col:
                grid_df[col] = grid_df[col].astype(np.float16)

        for i in [7, 14, 30, 60, 180]:
            grid_df[f'rolling_mean_{i}'] = grid_df.groupby(['id'])[self.TARGET].transform(lambda x: x.shift(shift_day).rolling(i).mean()).astype(np.float16)
            grid_df[f'rolling_std_{i}'] = grid_df.groupby(['id'])[self.TARGET].transform(lambda x: x.shift(shift_day).rolling(i).std()).astype(np.float16)

        for d_shift in [1, 7, 14]:
            for d_window in [7, 14, 30, 60]:
                col_name = f'rolling_mean_tmp_{d_shift}_{d_window}'
                grid_df[col_name] = grid_df.groupby(['id'])[self.TARGET].transform(lambda x: x.shift(d_shift).rolling(d_window).mean()).astype(np.float16)

        grid_df.to_pickle(self.config.lags_df_28)
        logger.info(f"Saved lags to {self.config.lags_df_28}")

    def create_mean_encodings(self):
        logger.info("Creating mean encodings...")
        grid_df = pd.read_pickle(self.config.grid_part_1)
        grid_df.loc[grid_df['d'] > (self.END_TRAIN - 28), self.TARGET] = np.nan
        base_cols = list(grid_df)

        icols = [
            ['state_id'], ['store_id'], ['cat_id'], ['dept_id'],
            ['state_id', 'cat_id'], ['state_id', 'dept_id'],
            ['store_id', 'cat_id'], ['store_id', 'dept_id'],
            ['item_id'], ['item_id', 'state_id'], ['item_id', 'store_id']
        ]

        for col in icols:
            col_name = '_' + '_'.join(col) + '_'
            grid_df[f'enc{col_name}mean'] = grid_df.groupby(col)[self.TARGET].transform('mean').astype(np.float16)
            grid_df[f'enc{col_name}std'] = grid_df.groupby(col)[self.TARGET].transform('std').astype(np.float16)

        keep_cols = [col for col in list(grid_df) if col not in base_cols]
        grid_df = grid_df[['id', 'd'] + keep_cols]

        grid_df.to_pickle(self.config.mean_encoding_df)
        logger.info(f"Saved mean encodings to {self.config.mean_encoding_df}")