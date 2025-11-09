# GPU Histogram + Reduction (CUDA)

This project implements:
- Parallel maximum and minimum reduction
- A 512-bin histogram
- Optimised use of shared memory and atomic operations

It loads a binary dataset of floats, finds the min/max values using multi-stage GPU reduction,
and computes histogram counts across evenly spaced bins.
