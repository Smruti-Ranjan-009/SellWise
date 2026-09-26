import gc
import time
import os
import pandas as pd
import numpy as np
import warnings
from SellWise import logger
from SellWise.entity.config_entity import EnsembleConfig

warnings.filterwarnings('ignore')

class EnsemblePredictor:
    """
    Combines the 6 individual model submission files (3 recursive + 3 non-recursive)
    by averaging their predictions across evaluation IDs, matching the original 
    3-1 Final Ensemble logic.
    """
    def __init__(self, config: EnsembleConfig):
        self.config = config

    def run(self):
        logger.info("Starting Ensemble Pipeline...")
        
        # Load sample submission and isolate evaluation IDs (rows 30490+)
        sample_sub = pd.read_csv(self.config.sample_submission_path)
        ids = pd.DataFrame({'id': sample_sub.iloc[self.config.eval_row_offset:]['id']})
        logger.info(f"Loaded evaluation target IDs: {len(ids)} rows")

        sub_dfs = []
        for file_name in self.config.submission_files:
            file_path = self.config.submission_dir / file_name
            
            # Strict file existence check
            if not file_path.exists():
                raise FileNotFoundError(f"Submission file not found at: {file_path}")

            logger.info(f"Loading submission: {file_path.name}")
            sub = pd.read_csv(file_path)
            
            # Merge with evaluation IDs on 'id' left join and set index
            sub_merged = ids.merge(sub, on='id', how='left').set_index('id')
            sub_dfs.append(sub_merged)

        # Calculate arithmetic mean across all 6 model predictions
        logger.info("Averaging 6 submission dataframes...")
        final_sub = sum(sub_dfs) / len(sub_dfs)

        # Export final submission CSV
        output_path = self.config.submission_dir / self.config.final_submission_filename
        final_sub.to_csv(output_path)
        logger.info(f"✅ Final ensemble successfully saved to: {output_path}")

        return final_sub