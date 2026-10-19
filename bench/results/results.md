| function | size | standard CPU tool | its time | widestats CPU | widestats GPU | GPU vs tool | GPU vs widestats CPU |
|---|---|---|---|---|---|---|---|
| `cor_pairs` | 5,000 variables x 100 samples | NumPy, tiled | 0.180 s | 0.068 s | 0.007 s | **25×** | 9.6× |
| `cor_pairs` | 20,000 variables x 100 samples | NumPy, tiled | 1.603 s | 0.571 s | 0.037 s | **43×** | 15.5× |
| `cor_pairs` | 50,000 variables x 100 samples | NumPy, tiled | 8.540 s | 3.307 s | 0.177 s | **48×** | 18.7× |
| `cor_perm` | 1,000 x 1,000 variables, 1,000 perms | NumPy BLAS loop | 5.677 s | 1.486 s | 0.092 s | **62×** | 16.2× |
| `cor_perm` | 3,000 x 3,000 variables, 1,000 perms | NumPy BLAS loop | 73 s | 17 s | 0.970 s | **75×** | 17.8× |
| `ttest_perm` | 20,000 genes x 100 samples, 1,000 perms | NumPy vectorised loop | 10 s | 0.458 s | 0.063 s | **163×** | 7.2× |
| `ttest_perm` | 20,000 genes x 100 samples, 10,000 perms | not run (too slow) | — | 4.174 s | 0.180 s | — | 23.1× |
| `granger` | 100 series x 1,000 times, lag 2 | statsmodels, pair by pair | 16 s | 0.007 s | 0.007 s | **2,251×** | 0.9× |
| `granger` | 300 series x 1,000 times, lag 2 | statsmodels, pair by pair | 145 s | 0.032 s | 0.021 s | **6,921×** | 1.5× |
| `granger` | 1,000 series x 1,000 times, lag 2 | not run (too slow) | — | 0.175 s | 0.128 s | — | 1.4× |
| `mutual_info` | 1,000 variables x 1,000 samples | NumPy, same algorithm | 2.569 s | 0.967 s | 0.037 s | **70×** | 26.4× |
| `mutual_info` | 3,000 variables x 1,000 samples | NumPy, same algorithm | 24 s | 8.300 s | 0.277 s | **86×** | 29.9× |
| `mutual_info` | 6,000 variables x 1,000 samples | NumPy, same algorithm | 76 s | 33 s | 1.072 s | **71×** | 30.5× |
