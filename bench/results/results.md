| function | size | standard CPU tool | its time | widestats CPU | widestats GPU | GPU vs tool | GPU vs widestats CPU |
|---|---|---|---|---|---|---|---|
| `cor_pairs` | 5,000 variables x 100 samples | NumPy, tiled | 0.181 s | 0.048 s | 0.007 s | **28×** | 7.3× |
| `cor_pairs` | 20,000 variables x 100 samples | NumPy, tiled | 1.266 s | 0.572 s | 0.038 s | **34×** | 15.2× |
| `cor_pairs` | 50,000 variables x 100 samples | NumPy, tiled | 8.434 s | 3.260 s | 0.166 s | **51×** | 19.7× |
| `cor_perm` | 1,000 x 1,000 variables, 1,000 perms | NumPy BLAS loop | 5.718 s | 1.486 s | 0.091 s | **63×** | 16.4× |
| `cor_perm` | 3,000 x 3,000 variables, 1,000 perms | NumPy BLAS loop | 73 s | 17 s | 0.969 s | **75×** | 17.8× |
| `ttest_perm` | 20,000 genes x 100 samples, 1,000 perms | NumPy vectorised loop | 11 s | 0.440 s | 0.064 s | **165×** | 6.9× |
| `ttest_perm` | 20,000 genes x 100 samples, 10,000 perms | not run (too slow) | — | 4.143 s | 0.179 s | — | 23.1× |
| `granger` | 100 series x 1,000 times, lag 2 | statsmodels, pair by pair | 16 s | 0.012 s | 0.013 s | **1,250×** | 1.0× |
| `granger` | 300 series x 1,000 times, lag 2 | statsmodels, pair by pair | 146 s | 0.073 s | 0.060 s | **2,426×** | 1.2× |
| `granger` | 1,000 series x 1,000 times, lag 2 | not run (too slow) | — | 0.811 s | 0.588 s | — | 1.4× |
| `mutual_info` | 1,000 variables x 1,000 samples | NumPy, same algorithm | 2.553 s | 0.969 s | 0.037 s | **69×** | 26.3× |
| `mutual_info` | 3,000 variables x 1,000 samples | NumPy, same algorithm | 23 s | 7.981 s | 0.275 s | **83×** | 29.0× |
| `mutual_info` | 6,000 variables x 1,000 samples | NumPy, same algorithm | 90 s | 31 s | 1.062 s | **85×** | 29.3× |
