"""
Sequential Single-Task (SST) utilities for SuTraN.

This package contains the pared-down encoder‑decoder architecture,
training/inference routines, and supporting utilities used to train
SuTraN on a single sequential head at a time: either activity suffix or
timestamp suffix prediction. It mirrors the polished SuTraN components
but strips out the multi-task heads so SST models can serve as strong
single-task baselines.
"""
