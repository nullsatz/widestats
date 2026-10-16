| function | size | standard CPU tool | its time | widestats CPU | widestats GPU | GPU vs tool | GPU vs widestats CPU |
|---|---|---|---|---|---|---|---|
| `cor_pairs` | 5,000 variables x 100 samples | NumPy, tiled | 0.191 s | 0.056 s | 0.008 s | **25×** | 7.3× |
| `cor_pairs` | 20,000 variables x 100 samples | NumPy, tiled | 1.690 s | 0.633 s | 0.039 s | **44×** | 16.4× |
| `cor_pairs` | 50,000 variables x 100 samples | NumPy, tiled | 6.928 s | 4.067 s | 0.187 s | **37×** | 21.7× |
| `cor_perm` | 1,000 x 1,000 variables, 1,000 perms | NumPy BLAS loop | 5.964 s | 1.280 s | 0.094 s | **63×** | 13.6× |
| `cor_perm` | 3,000 x 3,000 variables, 1,000 perms | NumPy BLAS loop | 63 s | 17 s | 0.947 s | **66×** | 18.3× |
| `ttest_perm` | 20,000 genes x 100 samples, 1,000 perms | NumPy vectorised loop | 11 s | 0.348 s | 0.064 s | **169×** | 5.4× |
| `ttest_perm` | 20,000 genes x 100 samples, 10,000 perms | not run (too slow) | — | 2.952 s | 0.186 s | — | 15.9× |
| `granger` | 100 series x 1,000 times, lag 2 | statsmodels, pair by pair | 16 s | 0.055 s | 0.060 s | **275×** | 0.9× |
| `granger` | 300 series x 1,000 times, lag 2 | statsmodels, pair by pair | 146 s | 0.485 s | 0.445 s | **329×** | 1.1× |
| `granger` | 1,000 series x 1,000 times, lag 2 | not run (too slow) | — | 5.228 s | 4.873 s | — | 1.1× |
| `mutual_info` | 1,000 variables x 1,000 samples | NumPy, same algorithm | 2.306 s | 0.879 s | 0.037 s | **62×** | 23.7× |
| `mutual_info` | 3,000 variables x 1,000 samples | NumPy, same algorithm | 19 s | 8.019 s | 0.266 s | **73×** | 30.1× |
| `mutual_info` | 6,000 variables x 1,000 samples | NumPy, same algorithm | 75 s | 34 s | 1.057 s | **71×** | 32.0× |
