| function | size | best CPU code | its time | widestats CPU | widestats GPU | GPU speedup |
|---|---|---|---|---|---|---|
| `cor_pairs` | 5,000 variables x 100 samples | NumPy corrcoef + search | 0.419 s | 0.046 s | 0.008 s | **52×** |
| `cor_pairs` | 20,000 variables x 100 samples | NumPy corrcoef + search | 6.991 s | 0.594 s | 0.038 s | **184×** |
| `cor_pairs` | 50,000 variables x 100 samples | CorALS cor_topk | 386 s | 3.047 s | 0.170 s | **2,272×** |
| `cor_perm` | 1,000 x 1,000 variables, 1000 perms | NumPy BLAS loop | 4.353 s | 1.296 s | 0.086 s | **51×** |
| `cor_perm` | 3,000 x 3,000 variables, 1000 perms | NumPy BLAS loop | 62 s | 15 s | 0.914 s | **68×** |
| `ttest_perm` | 20,000 genes x 100 samples, 1,000 perms | NumPy vectorised loop | 7.387 s | 0.390 s | 0.052 s | **142×** |
| `ttest_perm` | 20,000 genes x 100 samples, 10,000 perms | not run (too slow) | — | 3.671 s | 0.169 s | — |
| `granger` | 100 series x 1000 times, lag 2 | statsmodels, pair by pair | 16 s | 0.007 s | 0.007 s | **2,238×** |
| `granger` | 300 series x 1000 times, lag 2 | statsmodels, pair by pair | 142 s | 0.027 s | 0.024 s | **5,934×** |
| `granger` | 1,000 series x 1000 times, lag 2 | not run (too slow) | — | 0.181 s | 0.159 s | — |
| `mutual_info` | 1,000 variables x 1000 samples | NumPy, same algorithm | 2.184 s | 0.860 s | 0.031 s | **70×** |
| `mutual_info` | 3,000 variables x 1000 samples | NumPy, same algorithm | 18 s | 7.024 s | 0.258 s | **72×** |
| `mutual_info` | 6,000 variables x 1000 samples | NumPy, same algorithm | 73 s | 28 s | 1.008 s | **72×** |
