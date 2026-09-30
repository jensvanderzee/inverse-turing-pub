# -*- coding: utf-8 -*-
"""
Train real-data models 10-19, so a run can be split across cluster jobs.

Same configuration as realdata_train_invPDE.py; the model index fixes each run's
seed and output filename, so the split runs are identical to a single long run.
"""
#%%
from realdata_train_invPDE import REALDATA_MODEL_DIR, STEPS_PER_WEEK, train_models_real_data

if __name__ == "__main__":
    results, training_data = train_models_real_data(
        data_dir=r"./data",
        save_dir=REALDATA_MODEL_DIR,
        selected_sites=['b', 'i', 'c', 'e'],
        model_ids=list(range(10, 20)),
        ndvi_to_biomass_multiplier=1500,
        use_delta_loss=True,
        steps_per_week=STEPS_PER_WEEK,
        use_weekly_precip=True,
    )

# %%
