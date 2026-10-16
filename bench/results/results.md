| function | size | standard CPU tool | its time | widestats CPU | widestats GPU | GPU vs tool | GPU vs widestats CPU |
|---|---|---|---|---|---|---|---|
| `cor_pairs` | 5,000 variables x 100 samples | NumPy, tiled | 0.191 s | 0.057 s | 0.008 s | **23×** | 6.9× |
| `cor_pairs` | 20,000 variables x 100 samples | NumPy, tiled | 1.680 s | 0.669 s | 0.038 s | **44×** | 17.6× |
| `cor_pairs` | 50,000 variables x 100 samples | NumPy, tiled | 6.821 s | 3.451 s | 0.170 s | **40×** | 20.3× |
| `cor_perm` | 1,000 x 1,000 variables, 1,000 perms | NumPy BLAS loop | 5.950 s | 1.264 s | 0.093 s | **64×** | 13.5× |
| `cor_perm` | 3,000 x 3,000 variables, 1,000 perms | NumPy BLAS loop | 68 s | 21 s | 0.889 s | **77×** | 23.3× |
| `ttest_perm` | 20,000 genes x 100 samples, 1,000 perms | NumPy vectorised loop | 11 s | 0.428 s | 0.065 s | **169×** | 6.6× |
| `ttest_perm` | 20,000 genes x 100 samples, 10,000 perms | not run (too slow) | — | 4.486 s | 0.183 s | — | 24.5× |
| `granger` | 100 series x 1,000 times, lag 2 | statsmodels, pair by pair | 16 s | 0.009 s | 0.009 s | **1,800×** | 1.0× |
| `granger` | 300 series x 1,000 times, lag 2 | statsmodels, pair by pair | 147 s | 0.030 s | 0.028 s | **5,188×** | 1.1× |
| `granger` | 1,000 series x 1,000 times, lag 2 | not run (too slow) | — | 0.197 s | 0.177 s | — | 1.1× |
| `mutual_info` | 1,000 variables x 1,000 samples | NumPy, same algorithm | 2.316 s | 1.068 s | 0.037 s | **63×** | 28.9× |
| `mutual_info` | 3,000 variables x 1,000 samples | NumPy, same algorithm | 20 s | 9.581 s | 0.266 s | **76×** | 36.0× |
| `mutual_info` | 6,000 variables x 1,000 samples | NumPy, same algorithm | 76 s | 33 s | 1.084 s | **71×** | 30.0× |
