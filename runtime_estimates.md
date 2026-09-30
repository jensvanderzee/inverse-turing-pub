# Estimated single-run training time

Measured on: `CUDA: NVIDIA A100-SXM4-40GB  (Linux-6.6.113+-x86_64-with-glibc2.35)`  
Method: 20 timed epochs per experiment, extrapolated linearly to the target epoch count.

| Experiment | Epochs/run | Mean s/epoch | Estimated wall time | Peak GPU memory |
|---|---:|---:|---:|---:|
| invPDE synthetic, 1 site | 7500 | 0.966 | 2h 0m 45s | 435 MB |
| invPDE synthetic, 4 sites | 7500 | 3.795 | 7h 54m 22s | 1720 MB |
| RCNN synthetic, 4 sites | 1000 | 2.325 | 38m 44s | 25504 MB |
| invPDE real data, 4 sites | 7500 | 12.101 | 25h 12m 39s | 6844 MB |
