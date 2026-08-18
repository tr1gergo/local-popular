# Local Popularity Heuristics

This repository contains the NUMBA implementation and simulation artifacts for
the LocPop and LocStab heuristics studied in the NeurIPS 2025 paper "Clustering via Hedonic Games: New Concepts and Algorithms".

The four `*Numba.ipynb` notebooks contain experiments. Shared
code is in `BenchmarkNumbaExperiments.py`, `LocalPopularNumba.py`, and
`GraphFunctions.py`; generated data and plots are stored in `csv/` and
`figures/`.

To reproduce the experiments, install
`requirements-numba-experiments.txt` and run the notebooks from a fresh kernel
in this order: popular clustering, stable clustering, popular community
detection, and stable community detection. The production outputs use ten
paired repetitions with base seed `20260817`.
