#!/usr/bin/env python3
"""Compare two PPM files byte-by-byte; report sum/mean/max absolute difference.
Used to validate that mandelbrot_mpi (CPU) and mandelbrot_mpi_cuda (GPU)
produce visually equivalent images.
"""
import sys

def read_ppm(path):
    with open(path, 'rb') as f:
        magic = f.readline().strip()
        # skip comments
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()
        w, h = (int(x) for x in line.split())
        f.readline()  # max value
        return w, h, f.read()

def main():
    if len(sys.argv) != 3:
        print("usage: compare_ppm.py a.ppm b.ppm", file=sys.stderr)
        sys.exit(2)
    w1, h1, a = read_ppm(sys.argv[1])
    w2, h2, b = read_ppm(sys.argv[2])
    assert (w1, h1) == (w2, h2), f"size mismatch: {w1}x{h1} vs {w2}x{h2}"
    n = min(len(a), len(b))
    diff = sum(abs(a[i] - b[i]) for i in range(n))
    maxd = max(abs(a[i] - b[i]) for i in range(n)) if n else 0
    print(f"pixels: {w1}x{h1} = {w1*h1}")
    print(f"bytes:  {n}")
    print(f"sum-abs-diff: {diff}")
    print(f"mean-per-byte-diff: {diff/n:.4f}")
    print(f"max-byte-diff: {maxd}")

if __name__ == "__main__":
    main()
