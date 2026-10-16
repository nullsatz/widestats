| function | size | standard CPU tool | its time | widestats CPU | widestats GPU | GPU vs tool | GPU vs widestats CPU |
|---|---|---|---|---|---|---|---|
| `cor_pairs` | 5,000 variables x 100 samples | NumPy corrcoef + search | 0.355 s | 0.058 s | 0.008 s | **47×** | 7.7× |
| `cor_pairs` | 20,000 variables x 100 samples | NumPy corrcoef + search | 5.781 s | 0.697 s | 0.039 s | **148×** | 17.9× |
| `cor_pairs` | 50,000 variables x 100 samples | CorALS cor_topk | 394 s | 2.375 s | 0.186 s | **2,117×** | 12.8× |
| `cor_perm` | 1,000 x 1,000 variables, 1,000 perms | NumPy BLAS loop | 4.035 s | 1.347 s | 0.093 s | **43×** | 14.5× |
| `cor_perm` | 3,000 x 3,000 variables, 1,000 perms | NumPy BLAS loop | 39 s | 16 s | 0.887 s | **44×** | 18.5× |
| `ttest_perm` | 20,000 genes x 100 samples, 1,000 perms | NumPy vectorised loop | 11 s | 0.340 s | 0.066 s | **161×** | 5.1× |
| `ttest_perm` | 20,000 genes x 100 samples, 10,000 perms | not run (too slow) | — | 2.913 s | 0.185 s | — | 15.7× |
| `granger` | 100 series x 1,000 times, lag 2 | statsmodels, pair by pair | 16 s | 0.009 s | 0.009 s | **1,850×** | 1.0× |
| `granger` | 300 series x 1,000 times, lag 2 | statsmodels, pair by pair | 147 s | 0.030 s | 0.028 s | **5,203×** | 1.1× |
| `granger` | 1,000 series x 1,000 times, lag 2 | not run (too slow) | — | 0.196 s | 0.176 s | — | 1.1× |
| `mutual_info` | 1,000 variables x 1,000 samples | NumPy, same algorithm | 2.227 s | 0.794 s | 0.037 s | **60×** | 21.4× |
| `mutual_info` | 3,000 variables x 1,000 samples | NumPy, same algorithm | 19 s | 6.744 s | 0.266 s | **71×** | 25.3× |
| `mutual_info` | 6,000 variables x 1,000 samples | NumPy, same algorithm | 75 s | 37 s | 1.036 s | **72×** | 35.4× |
