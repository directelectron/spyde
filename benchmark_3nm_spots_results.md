# 3 nm scan -- spot-finding benchmark results

- dataset: `20241215_29639_movie_movie.mrc`  (300x300 nav, 256x256 uint16)
- beam stop: 2657 px masked
- scored on 120 spotty frames

Metrics: **bs_fp** = beam-stop false-positive rate (lower better); **fri** = Friedel inlier fraction (higher better); **n** = spots/frame off the stop; **t** = median ms/frame.

| config | bs_fp % | fri % | n_real | n | t (ms) |
|---|---:|---:|---:|---:|---:|
| nxcorr kr5 thr0.5 md5 s1.0 nomask | 0.0 | 6.9 | 1.6 | 24.0 | 9.99 |
| nxcorr kr1 thr0.4 md3 s0.8 mask+dil2 | 0.0 | 21.8 | 217.2 | 992.2 | 29.92 |
| nxcorr kr2 thr0.4 md3 s0.8 mask+dil2 | 0.0 | 8.8 | 29.6 | 324.7 | 15.00 |
| nxcorr kr3 thr0.4 md3 s0.8 mask+dil2 | 0.0 | 4.2 | 3.9 | 95.7 | 11.42 |
| nxcorr kr4 thr0.4 md3 s0.8 mask+dil2 | 0.0 | 8.6 | 3.4 | 42.0 | 10.43 |
| nxcorr kr5 thr0.4 md3 s0.8 mask+dil2 | 0.0 | 11.8 | 3.5 | 30.5 | 10.20 |
| nxcorr kr2 thr0.4 md3 s0.8 bs=none | 0.1 | 8.3 | 28.6 | 333.3 | 14.06 |
| nxcorr kr2 thr0.4 md3 s0.8 bs=mask | 0.4 | 8.4 | 29.0 | 332.2 | 14.87 |
| nxcorr kr2 thr0.4 md3 s0.8 bs=mask+dil2 | 0.0 | 8.8 | 29.6 | 324.7 | 14.48 |
| nxcorr kr2 thr0.25 md3 s0.8 mask+dil2 | 0.0 | 16.2 | 111.6 | 685.5 | 21.88 |
| nxcorr kr2 thr0.35 md3 s0.8 mask+dil2 | 0.0 | 10.7 | 46.3 | 422.2 | 16.67 |
| nxcorr kr2 thr0.45 md3 s0.8 mask+dil2 | 0.0 | 7.0 | 17.7 | 244.3 | 13.01 |
| nxcorr kr2 thr0.55 md3 s0.8 mask+dil2 | 0.0 | 4.7 | 6.2 | 125.2 | 11.46 |
| nxcorr kr2 thr0.65 md3 s0.8 mask+dil2 | 0.0 | 2.4 | 1.4 | 59.1 | 10.85 |
| log sig1.0 thr0.1 md3 | 67.4 | 9.6 | 1.9 | 19.3 | 7.49 |
| log sig1.0 thr0.2 md3 | 85.3 | 3.3 | 0.3 | 5.6 | 7.64 |
| log sig1.0 thr0.3 md3 | 91.0 | 1.9 | 0.1 | 2.7 | 7.47 |
| log sig1.4 thr0.1 md3 | 19.3 | 5.1 | 2.1 | 40.1 | 8.03 |
| log sig1.4 thr0.2 md3 | 28.2 | 8.0 | 1.9 | 24.4 | 8.07 |
| log sig1.4 thr0.3 md3 | 36.1 | 13.3 | 2.3 | 17.0 | 8.03 |
| log sig1.8 thr0.1 md3 | 6.7 | 7.7 | 3.0 | 38.9 | 8.17 |
| log sig1.8 thr0.2 md3 | 9.8 | 10.6 | 2.7 | 25.6 | 8.21 |
| log sig1.8 thr0.3 md3 | 12.7 | 14.6 | 2.8 | 19.2 | 8.20 |
| log sig2.2 thr0.1 md3 | 0.5 | 8.0 | 2.9 | 35.1 | 8.49 |
| log sig2.2 thr0.2 md3 | 0.7 | 8.8 | 2.1 | 23.7 | 8.41 |
| log sig2.2 thr0.3 md3 | 1.0 | 11.7 | 2.2 | 18.4 | 8.37 |
