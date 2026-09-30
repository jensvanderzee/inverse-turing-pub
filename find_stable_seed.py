# -*- coding: utf-8 -*-
"""
Find a seed for which biomass multipliers 750, 1500 and 3000 all produce
numerically stable training for at least 50 epochs on real satellite data.

Place this file next to realdata_train_invPDE.py and run:
    python find_stable_seed.py
"""

import torch
import math
from realdata_train_invPDE import (
    RealDataLoader,
    train_model_real_data,
)

# --- Configuration (match your original __main__ block) ---------------------
DATA_DIR        = r"./data"
SELECTED_SITES  = ['b', 'i', 'c', 'e']
STEPS_PER_WEEK  = 4
USE_WEEKLY_PRECIP = True
USE_DELTA_LOSS  = True
LEARNING_RATE   = 0.2

MULTIPLIERS     = [750, 1500, 3000]
TEST_EPOCHS     = 250        # number of epochs a seed must survive
MAX_SEED        = 1000      # search range  (0 .. MAX_SEED-1)
# -----------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# 1) Pre-load training data for every multiplier (done once)
training_datasets = {}
for mult in MULTIPLIERS:
    print(f"Loading data with ndvi_to_biomass_multiplier = {mult} ...")
    loader = RealDataLoader(
        DATA_DIR,
        selected_sites=SELECTED_SITES,
        device=device,
        use_weekly_precip=USE_WEEKLY_PRECIP,
    )
    training_datasets[mult] = loader.get_training_data(
        ndvi_to_biomass_multiplier=mult
    )
    print()

# 2) Search over seeds
print("=" * 60)
print(f"Testing seeds 0 - {MAX_SEED - 1}  |  "
      f"{TEST_EPOCHS} epochs  |  multipliers {MULTIPLIERS}")
print("=" * 60)

for seed in range(MAX_SEED):
    results = {}
    all_ok = True

    for mult in MULTIPLIERS:
        try:
            loss_history, _, _, _ = train_model_real_data(
                training_data=training_datasets[mult],
                num_epochs=TEST_EPOCHS,
                learning_rate=LEARNING_RATE,
                device=device,
                seed=seed,
                save_interval=TEST_EPOCHS + 1,   # skip saving snapshots
                steps_per_week=STEPS_PER_WEEK,
                use_delta_loss=USE_DELTA_LOSS,
            )
            # The loop breaks early on NaN/Inf, so fewer entries means crash
            stable = len(loss_history) == TEST_EPOCHS
            results[mult] = (stable, loss_history[-1] if loss_history else float("nan"))
        except Exception as e:
            results[mult] = (False, float("nan"))

        if not results[mult][0]:
            all_ok = False
            break          # no need to test remaining multipliers

    status = "  ".join(
        f"{m}: {'OK' if results.get(m, (False,))[0] else 'FAIL'}"
        for m in MULTIPLIERS
    )
    tag = "  <<<  STABLE" if all_ok else ""
    print(f"seed {seed:4d}  |  {status}{tag}")

    if all_ok:
        print("\n" + "=" * 60)
        print(f"FOUND STABLE SEED: {seed}")
        print("=" * 60)
        for m in MULTIPLIERS:
            print(f"  multiplier {m:5d}  ->  loss after {TEST_EPOCHS} epochs = "
                  f"{results[m][1]:.6f}")
        break
else:
    print(f"\nNo stable seed found in range 0 - {MAX_SEED - 1}.")
