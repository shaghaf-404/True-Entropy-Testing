"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              TRUE ENTROPY TESTING — NIST SP 800-22 Test Suite               ║
║    QRNG (Quantum) vs TRNG (/dev/urandom) vs PRNG (MT19937/LCG/Xorshift)    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Implements 15 NIST SP 800-22 statistical randomness tests and compares:
  1. QRNG_sim  — Simulated quantum (photon shot noise + vacuum fluctuations)
  2. TRNG      — /dev/urandom (Linux CSPRNG, hardware-seeded)
  3. MT19937   — Mersenne Twister (Python's random module, gold-standard PRNG)
  4. LCG       — Linear Congruential Generator (weak classical PRNG)
  5. Xorshift  — Xorshift64 (fast modern PRNG)

NIST Tests Implemented (SP 800-22 Rev 1a):
  01. Monobit Frequency Test
  02. Block Frequency Test
  03. Runs Test
  04. Longest Run of Ones in a Block
  05. Binary Matrix Rank Test
  06. Discrete Fourier Transform (Spectral) Test
  07. Non-overlapping Template Matching Test
  08. Overlapping Template Matching Test
  09. Maurer's Universal Statistical Test
  10. Linear Complexity Test
  11. Serial Test
  12. Approximate Entropy Test
  13. Cumulative Sums Test
  14. Random Excursions Test
  15. Random Excursions Variant Test
"""

import os
import sys
import time
import math
import random
import struct
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy import special, stats
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
N_BITS        = 500_000   # bits per source (NIST recommends ≥1M)
BLOCK_SIZE_M  = 128         # block size for block frequency test
SIGNIFICANCE  = 0.01        # α = 0.01 significance level

# ─── Source Definitions ───────────────────────────────────────────────────────
SOURCES = {
    "QRNG_sim":  {"label": "QRNG (Simulated Quantum)",    "color": "#4FC3F7", "marker": "o"},
    "TRNG":      {"label": "TRNG (/dev/urandom)",         "color": "#81C784", "marker": "s"},
    "MT19937":   {"label": "PRNG (Mersenne Twister)",     "color": "#FFA726", "marker": "^"},
    "LCG":       {"label": "PRNG (LCG — Weak)",           "color": "#EF5350", "marker": "D"},
    "Xorshift64":{"label": "PRNG (Xorshift64)",           "color": "#CE93D8", "marker": "P"},
}

# ─── Data Generators ──────────────────────────────────────────────────────────

def gen_qrng_simulated(n_bits: int) -> np.ndarray:
    """
    Simulates a photon-counting QRNG.
    Physical model: At a 50/50 beam splitter, each photon has exactly 50%
    probability of transmission or reflection — a fundamentally quantum event
    (Born rule probability). Shot noise follows Poisson statistics.
    We model arrival-time quantisation: sample Poisson-distributed photon
    arrival counts and threshold to binary.
    Additional vacuum fluctuation layer adds Gaussian noise before thresholding.
    """
    rng = np.random.default_rng(int.from_bytes(os.urandom(8), 'big'))
    # Photon arrivals: Poisson(λ=1), threshold at mean
    photon_counts = rng.poisson(lam=1.0, size=n_bits)
    # Vacuum fluctuation noise (Gaussian quadrature noise)
    vacuum_noise  = rng.normal(0, 0.5, n_bits)
    # Combined signal: threshold at 0.5
    signal = photon_counts + vacuum_noise
    bits   = (signal > np.median(signal)).astype(np.uint8)
    return bits

def gen_trng_dev_urandom(n_bits: int) -> np.ndarray:
    """
    Reads from Linux /dev/urandom.
    Since Linux 4.8, /dev/urandom uses ChaCha20-based CSPRNG seeded from
    the kernel entropy pool (hardware events: interrupts, disk I/O, RDRAND).
    On this system RDRAND (hardware RNG) is available (confirmed in cpuinfo).
    """
    n_bytes = (n_bits + 7) // 8
    raw     = os.urandom(n_bytes)
    arr     = np.frombuffer(raw, dtype=np.uint8)
    bits    = np.unpackbits(arr)[:n_bits]
    return bits

def gen_mt19937(n_bits: int) -> np.ndarray:
    """
    Mersenne Twister (MT19937) — Python's default random module.
    Period 2^19937-1, passes most statistical tests, but is NOT
    cryptographically secure (state can be reconstructed from 624 outputs).
    """
    rng   = np.random.default_rng(np.random.MT19937(seed=42))
    n_u32 = (n_bits + 31) // 32
    vals  = rng.integers(0, 2**32, size=n_u32, dtype=np.uint32)
    bits  = np.unpackbits(vals.view(np.uint8))[:n_bits]
    return bits

def gen_lcg(n_bits: int) -> np.ndarray:
    """
    Linear Congruential Generator: X_{n+1} = (a*X_n + c) mod m
    Uses ANSI C parameters: a=1103515245, c=12345, m=2^31
    Known to have poor statistical properties — short period in low bits,
    detectable linear structure. Used to demonstrate PRNG weakness.
    """
    a, c, m = 1103515245, 12345, 2**31
    x       = 12345678  # seed
    bits    = np.empty(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        x       = (a * x + c) & (m - 1)
        bits[i] = (x >> 15) & 1  # use bit 15 (higher bits better in LCG)
    return bits

def gen_xorshift64(n_bits: int) -> np.ndarray:
    """
    Xorshift64: fast non-linear shift-register PRNG.
    X ^= X << 13; X ^= X >> 7; X ^= X << 17
    Good statistical properties, passes most Diehard tests,
    but still deterministic from seed — NOT cryptographically secure.
    """
    x    = np.uint64(123456789012345)
    bits = np.empty(n_bits, dtype=np.uint8)
    i    = 0
    while i < n_bits:
        x ^= np.uint64(x << np.uint64(13))
        x ^= np.uint64(x >> np.uint64(7))
        x ^= np.uint64(x << np.uint64(17))
        word = int(x)
        for b in range(min(64, n_bits - i)):
            bits[i + b] = (word >> b) & 1
        i += 64
    return bits[:n_bits]

GENERATORS = {
    "QRNG_sim":   gen_qrng_simulated,
    "TRNG":       gen_trng_dev_urandom,
    "MT19937":    gen_mt19937,
    "LCG":        gen_lcg,
    "Xorshift64": gen_xorshift64,
}

# ─── NIST SP 800-22 Tests ─────────────────────────────────────────────────────

@dataclass
class TestResult:
    name:       str
    p_value:    float
    passed:     bool
    statistic:  float = 0.0
    details:    str   = ""

def bits_to_pm1(bits: np.ndarray) -> np.ndarray:
    """Convert {0,1} bits to {-1,+1} for sum-based tests."""
    return 2 * bits.astype(np.int64) - 1

# ── Test 1: Monobit Frequency ─────────────────────────────────────────────────
def test_monobit(bits: np.ndarray) -> TestResult:
    n  = len(bits)
    Sn = np.abs(np.sum(bits_to_pm1(bits)))
    s_obs = Sn / math.sqrt(n)
    p = math.erfc(s_obs / math.sqrt(2))
    return TestResult("01 Monobit Frequency", p, p >= SIGNIFICANCE, s_obs,
                      f"Σ={np.sum(bits)}, n={n}, S_obs={s_obs:.4f}")

# ── Test 2: Block Frequency ───────────────────────────────────────────────────
def test_block_frequency(bits: np.ndarray, M: int = 128) -> TestResult:
    n      = len(bits)
    N      = n // M
    blocks = bits[:N*M].reshape(N, M)
    props  = blocks.mean(axis=1)
    chi2   = 4 * M * np.sum((props - 0.5)**2)
    p      = special.gammaincc(N / 2, chi2 / 2)
    return TestResult("02 Block Frequency", p, p >= SIGNIFICANCE, chi2,
                      f"N={N} blocks of M={M}, χ²={chi2:.4f}")

# ── Test 3: Runs ──────────────────────────────────────────────────────────────
def test_runs(bits: np.ndarray) -> TestResult:
    n    = len(bits)
    pi   = np.mean(bits)
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return TestResult("03 Runs", 0.0, False, 0.0,
                          f"Pre-condition failed: π={pi:.4f}")
    Vn   = 1 + np.sum(bits[:-1] != bits[1:])
    num  = abs(Vn - 2*n*pi*(1-pi))
    den  = 2 * math.sqrt(2*n) * pi * (1-pi)
    p    = math.erfc(num / den)
    return TestResult("03 Runs", p, p >= SIGNIFICANCE, Vn,
                      f"V_n={Vn}, π={pi:.5f}")

# ── Test 4: Longest Run of Ones in a Block ────────────────────────────────────
def test_longest_run(bits: np.ndarray) -> TestResult:
    n = len(bits)
    if n < 128:
        return TestResult("04 Longest Run", 0.0, False, 0, "n too small")
    # Use M=8 (n<6272), M=128 (n<750000), M=10000 otherwise
    if n < 6272:
        M, K, pi_vals, V_vals = 8, 3, [0.2148,0.3672,0.2305,0.1875], [1,2,3,4]
    elif n < 750000:
        M, K, pi_vals, V_vals = 128, 5, [0.1174,0.2430,0.2493,0.1752,0.1027,0.1124], [4,5,6,7,8,9]
    else:
        M, K, pi_vals, V_vals = 10000, 6, [0.0882,0.2092,0.2483,0.1933,0.1208,0.0675,0.0727], [10,11,12,13,14,15,16]

    N      = n // M
    blocks = bits[:N*M].reshape(N, M)
    freq   = np.zeros(K+1, dtype=int)
    for block in blocks:
        # Find longest run of 1s
        max_run = cur = 0
        for b in block:
            cur    = cur + 1 if b else 0
            max_run = max(max_run, cur)
        v = min(max_run, V_vals[-1])
        idx = V_vals.index(v) if v in V_vals else (len(V_vals)-1 if v >= V_vals[-1] else 0)
        freq[idx] += 1

    # Chi-squared
    chi2 = sum((freq[i] - N*pi_vals[i])**2 / (N*pi_vals[i])
                for i in range(min(len(freq), len(pi_vals))))
    p    = special.gammaincc(K/2, chi2/2)
    return TestResult("04 Longest Run of Ones", p, p >= SIGNIFICANCE, chi2,
                      f"N={N}, M={M}, χ²={chi2:.4f}")

# ── Test 5: Binary Matrix Rank ────────────────────────────────────────────────
def _gf2_rank(matrix: np.ndarray) -> int:
    """Gaussian elimination over GF(2)."""
    m     = matrix.copy().astype(np.uint8)
    rows, cols = m.shape
    rank  = 0
    for col in range(cols):
        pivot = None
        for row in range(rank, rows):
            if m[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        m[[rank, pivot]] = m[[pivot, rank]]
        for row in range(rows):
            if row != rank and m[row, col]:
                m[row] ^= m[rank]
        rank += 1
    return rank

def test_matrix_rank(bits: np.ndarray, M: int = 32, Q: int = 32) -> TestResult:
    n = len(bits)
    N = n // (M * Q)
    if N < 1:
        return TestResult("05 Binary Matrix Rank", 0.0, False, 0, "n too small")
    F_M   = F_M1 = F_rest = 0
    for i in range(N):
        block  = bits[i*M*Q:(i+1)*M*Q].reshape(M, Q)
        rank   = _gf2_rank(block)
        if rank == M:     F_M   += 1
        elif rank == M-1: F_M1  += 1
        else:             F_rest += 1
    p0, p1, p2 = 0.2888, 0.5776, 0.1336
    chi2 = ((F_M - p0*N)**2/(p0*N) + (F_M1 - p1*N)**2/(p1*N) +
            (F_rest - p2*N)**2/(p2*N))
    p    = math.exp(-chi2/2)
    return TestResult("05 Binary Matrix Rank", p, p >= SIGNIFICANCE, chi2,
                      f"N={N} matrices, F_M={F_M}, F_M-1={F_M1}")

# ── Test 6: Discrete Fourier Transform (Spectral) ────────────────────────────
def test_dft(bits: np.ndarray) -> TestResult:
    n    = len(bits)
    x    = bits_to_pm1(bits[:n if n % 2 == 0 else n-1]).astype(float)
    f    = np.fft.fft(x)
    mag  = np.abs(f[:len(x)//2])
    T    = math.sqrt(math.log(1/0.05) * len(x))
    N1   = np.sum(mag < T)
    N0   = 0.95 * len(x) / 2
    d    = (N1 - N0) / math.sqrt(len(x) * 0.95 * 0.05 / 4)
    p    = math.erfc(abs(d) / math.sqrt(2))
    return TestResult("06 DFT / Spectral", p, p >= SIGNIFICANCE, d,
                      f"N1={N1}, N0={N0:.1f}, d={d:.4f}")

# ── Test 7: Non-overlapping Template Matching ─────────────────────────────────
def test_non_overlapping_template(bits: np.ndarray,
                                  template: list = None) -> TestResult:
    if template is None:
        template = [0, 0, 1, 0, 0, 1, 0, 1, 1, 1]  # NIST example template
    m    = len(template)
    n    = len(bits)
    M    = 1000
    N    = n // M
    tmpl = np.array(template)
    mu   = (M - m + 1) / 2**m
    sig2 = M * (1/2**m - (2*m-1)/2**(2*m))
    counts = []
    for i in range(N):
        block = bits[i*M:(i+1)*M]
        # Vectorised: find all matching positions then count non-overlapping
        windows = np.lib.stride_tricks.sliding_window_view(block, m)
        match_pos = np.where(np.all(windows == tmpl, axis=1))[0]
        cnt  = 0
        prev = -m
        for pos in match_pos:
            if pos >= prev + m:
                cnt += 1
                prev = pos
        counts.append(cnt)
    chi2 = sum((c - mu)**2 / sig2 for c in counts)
    p    = special.gammaincc(N/2, chi2/2)
    return TestResult("07 Non-overlapping Template", p, p >= SIGNIFICANCE, chi2,
                      f"N={N}, μ={mu:.4f}, σ²={sig2:.4f}")

# ── Test 8: Overlapping Template Matching ────────────────────────────────────
def test_overlapping_template(bits: np.ndarray) -> TestResult:
    n    = len(bits)
    m    = 9
    M    = 1032
    N    = n // M
    K    = 5
    pi   = [0.364091, 0.185659, 0.139381, 0.100571, 0.070432, 0.139865]
    tmpl = np.ones(m, dtype=np.uint8)
    nu   = np.zeros(K+1, dtype=int)
    for i in range(N):
        block = bits[i*M:(i+1)*M]
        # Vectorised: check all windows at once using stride tricks
        windows = np.lib.stride_tricks.sliding_window_view(block, m)
        cnt = int(np.sum(np.all(windows == tmpl, axis=1)))
        nu[min(cnt, K)] += 1
    lam  = (M - m + 1) / 2**m
    eta  = lam / 2
    chi2 = sum((nu[k] - N*pi[k])**2 / (N*pi[k]) for k in range(K+1) if N*pi[k] > 0)
    p    = special.gammaincc(K/2, chi2/2)
    return TestResult("08 Overlapping Template", p, p >= SIGNIFICANCE, chi2,
                      f"N={N}, λ={lam:.4f}, η={eta:.4f}")

# ── Test 9: Maurer's Universal Statistical Test ───────────────────────────────
def test_universal(bits: np.ndarray, L: int = 7, Q: int = 1280) -> TestResult:
    n    = len(bits)
    K    = n // L - Q
    if K < 0:
        return TestResult("09 Universal (Maurer)", 0.0, False, 0, "n too small")
    # Build table of last occurrence positions
    table = {}
    for i in range(Q):
        block = tuple(bits[i*L:(i+1)*L])
        table[block] = i + 1
    fn = 0.0
    for i in range(Q, Q + K):
        block = tuple(bits[i*L:(i+1)*L])
        if block in table:
            fn += math.log2(i + 1 - table[block])
        table[block] = i + 1
    fn  /= K
    # Expected value and variance (from NIST SP 800-22 Table 5)
    ev  = {6:5.2177052, 7:6.1962507, 8:7.1836656, 9:8.1764248, 10:9.1723243}
    var = {6:2.954, 7:3.125, 8:3.238, 9:3.311, 10:3.356}
    exp_val = ev.get(L, 6.0)
    sigma   = math.sqrt(var.get(L, 3.0) / K)
    p       = math.erfc(abs(fn - exp_val) / (math.sqrt(2) * sigma))
    return TestResult("09 Universal (Maurer)", p, p >= SIGNIFICANCE, fn,
                      f"f_n={fn:.5f}, expected={exp_val:.5f}")

# ── Test 10: Linear Complexity ────────────────────────────────────────────────
def _berlekamp_massey(bits):
    """Berlekamp-Massey algorithm for linear complexity."""
    n  = len(bits)
    c  = np.zeros(n+1, dtype=int); c[0] = 1
    b  = np.zeros(n+1, dtype=int); b[0] = 1
    L  = 0; m = 1; db = 1
    for N_ in range(n):
        d = int(bits[N_])
        for i in range(1, L+1):
            d ^= c[i] & bits[N_-i]
        d &= 1
        if d == 0:
            m += 1
        elif 2*L <= N_:
            t = c.copy()
            for i in range(m, n+1-m):
                c[i] ^= (db * b[i-m]) & 1
            L = N_ + 1 - L; b = t; db = d; m = 1
        else:
            for i in range(m, n+1-m):
                c[i] ^= (db * b[i-m]) & 1
            m += 1
    return L

def test_linear_complexity(bits: np.ndarray, M: int = 500) -> TestResult:
    n  = len(bits)
    N  = min(n // M, 50)  # cap at 50 blocks for performance
    K  = 6
    pi = [0.010417, 0.03125, 0.125, 0.5, 0.25, 0.0625, 0.020833]
    mu = M/2 + (9 + (-1)**(M+1)) / 36 - (M/3 + 2/9) / 2**M
    nu = np.zeros(K+1, dtype=int)
    for i in range(N):
        block = bits[i*M:(i+1)*M]
        L_i   = _berlekamp_massey(block)
        t     = (-1)**M * (L_i - mu) + 2/9
        if   t <= -2.5: nu[0] += 1
        elif t <= -1.5: nu[1] += 1
        elif t <= -0.5: nu[2] += 1
        elif t <=  0.5: nu[3] += 1
        elif t <=  1.5: nu[4] += 1
        elif t <=  2.5: nu[5] += 1
        else:           nu[6] += 1
    chi2 = sum((nu[k] - N*pi[k])**2 / (N*pi[k]) for k in range(K+1) if N*pi[k] > 0)
    p    = special.gammaincc(K/2, chi2/2)
    return TestResult("10 Linear Complexity", p, p >= SIGNIFICANCE, chi2,
                      f"N={N} blocks of M={M}")

# ── Test 11: Serial Test ──────────────────────────────────────────────────────
def _psi_sq(bits: np.ndarray, m: int) -> float:
    """Fast version using integer pattern hashing on a 50k subset."""
    n_use = min(len(bits), 50_000)
    b     = bits[:n_use]
    n     = n_use
    counts = Counter()
    for i in range(n):
        val = 0
        for j in range(m):
            val = (val << 1) | int(b[(i + j) % n])
        counts[val] += 1
    return (2**m / n) * sum(v**2 for v in counts.values()) - n

def test_serial(bits: np.ndarray, m: int = 16) -> TestResult:
    psi_m   = _psi_sq(bits, m)
    psi_m1  = _psi_sq(bits, m-1)
    psi_m2  = _psi_sq(bits, m-2)
    dpsi2   = psi_m  - psi_m1
    d2psi2  = psi_m  - 2*psi_m1 + psi_m2
    p1 = special.gammaincc(2**(m-2), dpsi2/2)
    p2 = special.gammaincc(2**(m-3), d2psi2/2)
    p  = min(p1, p2)
    return TestResult("11 Serial Test", p, p >= SIGNIFICANCE, d2psi2,
                      f"ψ²_m={psi_m:.2f}, Δψ²={dpsi2:.2f}, Δ²ψ²={d2psi2:.2f}")

# ── Test 12: Approximate Entropy ─────────────────────────────────────────────
def test_approx_entropy(bits: np.ndarray, m: int = 10) -> TestResult:
    n_use = min(len(bits), 50_000)
    n     = n_use
    b     = bits[:n]
    def phi(m_):
        counts = Counter()
        for i in range(n):
            val = 0
            for j in range(m_):
                val = (val << 1) | int(b[(i+j) % n])
            counts[val] += 1
        return sum(c/n * math.log(c/n) for c in counts.values() if c > 0)
    apEn  = phi(m) - phi(m+1)
    chi2  = 2 * n * (math.log(2) - apEn)
    p     = special.gammaincc(2**(m-1), chi2/2)
    return TestResult("12 Approximate Entropy", p, p >= SIGNIFICANCE, apEn,
                      f"ApEn({m})={apEn:.6f}")

# ── Test 13: Cumulative Sums ──────────────────────────────────────────────────
def test_cumulative_sums(bits: np.ndarray) -> TestResult:
    n  = len(bits)
    x  = bits_to_pm1(bits)
    S  = np.cumsum(x)
    z  = np.max(np.abs(S))
    # Forward and backward
    S2 = np.cumsum(x[::-1])
    z2 = np.max(np.abs(S2))
    z  = max(z, z2)
    def sum_term(z, n):
        k_range = range(int((-n/z+1)/4), int((n/z-1)/4)+1)
        s = sum(stats.norm.cdf((4*k+1)*z/math.sqrt(n)) -
                stats.norm.cdf((4*k-1)*z/math.sqrt(n)) for k in k_range)
        k_range2 = range(int((-n/z-3)/4), int((n/z-1)/4)+1)
        s -= sum(stats.norm.cdf((4*k+3)*z/math.sqrt(n)) -
                 stats.norm.cdf((4*k+1)*z/math.sqrt(n)) for k in k_range2)
        return 1 - s
    try:
        p = sum_term(z, n)
    except:
        p = 0.0
    p = max(0.0, min(1.0, p))
    return TestResult("13 Cumulative Sums", p, p >= SIGNIFICANCE, z,
                      f"max|S_k|={z}")

# ── Test 14: Random Excursions ────────────────────────────────────────────────
def test_random_excursions(bits: np.ndarray) -> TestResult:
    x     = bits_to_pm1(bits)
    S     = np.concatenate([[0], np.cumsum(x), [0]])
    # Find cycles (zero crossings)
    zeros = np.where(S == 0)[0]
    if len(zeros) < 2:
        return TestResult("14 Random Excursions", 0.0, False, 0, "No cycles found")
    J = len(zeros) - 1
    states = [-4,-3,-2,-1,1,2,3,4]
    # pi probabilities from NIST
    pi_table = {
        1: [0.5000,0.2500,0.1250,0.0625,0.0313,0.0313],
        2: [0.7500,0.0625,0.0469,0.0352,0.0264,0.0791],
        3: [0.8333,0.0278,0.0231,0.0193,0.0161,0.0804],
        4: [0.8750,0.0156,0.0137,0.0120,0.0105,0.0733],
    }
    min_p = 1.0
    for x_state in states:
        s = abs(x_state)
        if s not in pi_table: continue
        pi  = pi_table[s]
        nu  = np.zeros(6, dtype=int)
        for i in range(J):
            seg   = S[zeros[i]:zeros[i+1]+1]
            count = np.sum(seg == x_state)
            nu[min(count, 5)] += 1
        chi2 = sum((nu[k] - J*pi[k])**2 / (J*pi[k])
                   for k in range(6) if J*pi[k] > 0)
        p_x  = special.gammaincc(5/2, chi2/2)
        min_p = min(min_p, p_x)
    return TestResult("14 Random Excursions", min_p, min_p >= SIGNIFICANCE, min_p,
                      f"J={J} cycles, worst state p={min_p:.4f}")

# ── Test 15: Random Excursions Variant ───────────────────────────────────────
def test_random_excursions_variant(bits: np.ndarray) -> TestResult:
    x     = bits_to_pm1(bits)
    S     = np.concatenate([[0], np.cumsum(x), [0]])
    zeros = np.where(S == 0)[0]
    if len(zeros) < 2:
        return TestResult("15 RE Variant", 0.0, False, 0, "No cycles found")
    J     = len(zeros) - 1
    min_p = 1.0
    for x_state in range(-9, 10):
        if x_state == 0: continue
        xi   = np.sum(S == x_state)
        p    = math.erfc(abs(xi - J) / math.sqrt(2 * J * (4*abs(x_state)-2)))
        min_p = min(min_p, p)
    return TestResult("15 RE Variant", min_p, min_p >= SIGNIFICANCE, min_p,
                      f"J={J}, worst state p={min_p:.4f}")

# ─── Entropy Metrics ─────────────────────────────────────────────────────────

def shannon_entropy(bits: np.ndarray) -> float:
    """Shannon entropy in bits per bit (max = 1.0)."""
    p1 = np.mean(bits)
    p0 = 1 - p1
    if p1 == 0 or p0 == 0: return 0.0
    return -(p0 * math.log2(p0) + p1 * math.log2(p1))

def min_entropy(bits: np.ndarray, block: int = 8) -> float:
    """Min-entropy H_∞ = -log2(max probability over all m-bit patterns)."""
    n       = len(bits) // block * block
    arr     = bits[:n].reshape(-1, block)
    vals    = np.packbits(arr, axis=1).flatten()
    counts  = Counter(vals)
    p_max   = max(counts.values()) / len(vals)
    return -math.log2(p_max)

def compression_ratio(bits: np.ndarray) -> float:
    """Attempt gzip compression on bit stream. True random → ratio ≈ 1.0."""
    import gzip, io
    raw  = np.packbits(bits).tobytes()
    buf  = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9) as f:
        f.write(raw)
    return len(buf.getvalue()) / len(raw)

def autocorrelation(bits: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Normalized autocorrelation for lags 1..max_lag."""
    x    = bits_to_pm1(bits).astype(float)
    n    = len(x)
    var  = np.var(x)
    if var == 0: return np.zeros(max_lag)
    ac = np.array([np.mean(x[lag:] * x[:n-lag]) / var for lag in range(1, max_lag+1)])
    return ac

# ─── Run all tests ────────────────────────────────────────────────────────────

ALL_TESTS = [
    test_monobit,
    test_block_frequency,
    test_runs,
    test_longest_run,
    test_matrix_rank,
    test_dft,
    test_non_overlapping_template,
    test_overlapping_template,
    test_universal,
    test_linear_complexity,
    lambda b: test_serial(b, m=10),        # Reduced m for speed
    lambda b: test_approx_entropy(b, m=8), # Reduced m for speed
    test_cumulative_sums,
    test_random_excursions,
    test_random_excursions_variant,
]

TEST_NAMES = [
    "01 Monobit Frequency", "02 Block Frequency", "03 Runs",
    "04 Longest Run", "05 Binary Matrix Rank", "06 DFT/Spectral",
    "07 Non-overlapping Template", "08 Overlapping Template",
    "09 Universal (Maurer)", "10 Linear Complexity", "11 Serial",
    "12 Approx Entropy", "13 Cumulative Sums", "14 Random Excursions",
    "15 RE Variant"
]

def run_all_tests(bits: np.ndarray, source_name: str) -> list[TestResult]:
    results = []
    for i, test_fn in enumerate(ALL_TESTS):
        try:
            r = test_fn(bits)
        except Exception as e:
            r = TestResult(TEST_NAMES[i], 0.0, False, 0.0, f"Error: {e}")
        results.append(r)
    return results

# ─── Visualization ────────────────────────────────────────────────────────────

def generate_report(all_results: dict, all_bits: dict, entropy_metrics: dict):
    C_BG    = "#0D1117"
    C_PANEL = "#161B22"
    C_TEXT  = "#E6EDF3"
    C_MUTED = "#8B949E"
    C_GRID  = "#21262D"
    C_PASS  = "#2EA043"
    C_FAIL  = "#DA3633"
    C_WARN  = "#D29922"

    fig = plt.figure(figsize=(22, 18), facecolor=C_BG)
    gs  = GridSpec(4, 3, figure=fig,
                   hspace=0.52, wspace=0.32,
                   top=0.92, bottom=0.05, left=0.06, right=0.97)

    def ax_style(ax, title=""):
        ax.set_facecolor(C_PANEL)
        for spine in ax.spines.values(): spine.set_edgecolor(C_GRID)
        ax.tick_params(colors=C_MUTED, labelsize=8)
        ax.xaxis.label.set_color(C_MUTED)
        ax.yaxis.label.set_color(C_MUTED)
        if title: ax.set_title(title, color=C_TEXT, fontsize=9.5, fontweight='bold', pad=7)
        ax.grid(True, color=C_GRID, lw=0.5, alpha=0.7)

    # Title
    fig.text(0.5, 0.958, "True Entropy Testing — NIST SP 800-22 Statistical Test Suite",
             ha='center', color=C_TEXT, fontsize=15, fontweight='bold')
    fig.text(0.5, 0.935,
             "QRNG (Simulated Quantum)  •  TRNG (/dev/urandom)  •  MT19937  •  LCG  •  Xorshift64",
             ha='center', color=C_MUTED, fontsize=9)

    src_names  = list(SOURCES.keys())
    src_colors = [SOURCES[s]["color"] for s in src_names]
    src_labels = [SOURCES[s]["label"] for s in src_names]

    # ── Panel 1: Pass/Fail heatmap ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax_style(ax1, "① NIST SP 800-22 — Pass/Fail Heatmap (15 Tests × 5 Sources)")
    short_names = [t.split(" ",1)[1][:28] for t in TEST_NAMES]
    matrix = np.zeros((len(src_names), len(TEST_NAMES)))
    for j, src in enumerate(src_names):
        for i, r in enumerate(all_results[src]):
            matrix[j, i] = r.p_value

    # Use custom colormap: red=fail, green=pass, yellow=marginal
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('nist', ['#DA3633','#D29922','#2EA043'],N=256)
    im = ax1.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1,
                    interpolation='nearest')
    ax1.axhline(y=-0.5+len(src_names), color=C_GRID, lw=0.5)
    ax1.set_xticks(range(len(TEST_NAMES)))
    ax1.set_xticklabels([f"T{i+1:02d}" for i in range(len(TEST_NAMES))],
                        fontsize=7, color=C_MUTED)
    ax1.set_yticks(range(len(src_names)))
    ax1.set_yticklabels(src_labels, fontsize=7.5, color=C_TEXT)
    # Draw significance threshold line
    ax1.axvline(x=-0.5, color=C_GRID, lw=0.5)
    # Annotate p-values
    for j in range(len(src_names)):
        for i in range(len(TEST_NAMES)):
            pv   = matrix[j, i]
            clr  = "white" if pv < 0.3 else "black"
            mark = "✓" if pv >= SIGNIFICANCE else "✗"
            ax1.text(i, j, mark, ha='center', va='center', fontsize=7,
                     color=clr, fontweight='bold')
    plt.colorbar(im, ax=ax1, fraction=0.015, pad=0.01,
                 label="p-value").ax.tick_params(labelsize=7, colors=C_MUTED)

    # ── Panel 2: Pass rate bar chart ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax_style(ax2, "② Pass Rate (of 15 Tests)")
    pass_rates = [sum(1 for r in all_results[s] if r.passed) / len(TEST_NAMES) * 100
                  for s in src_names]
    bars = ax2.barh(range(len(src_names)), pass_rates, color=src_colors,
                    edgecolor=C_GRID, height=0.6)
    ax2.set_yticks(range(len(src_names)))
    ax2.set_yticklabels(src_labels, fontsize=7.5, color=C_TEXT)
    ax2.set_xlabel("% Tests Passed", fontsize=8)
    ax2.set_xlim(0, 105)
    ax2.axvline(x=100, color=C_GRID, lw=1, ls='--')
    for bar, rate in zip(bars, pass_rates):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{rate:.0f}%", va='center', color=C_TEXT, fontsize=8.5, fontweight='bold')

    # ── Panel 3: p-value distribution (box plots) ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax_style(ax3, "③ p-value Distribution per Source")
    pval_data = [[r.p_value for r in all_results[s]] for s in src_names]
    bp = ax3.boxplot(pval_data, vert=True, patch_artist=True,
                     medianprops=dict(color='white', lw=2),
                     whiskerprops=dict(color=C_MUTED),
                     capprops=dict(color=C_MUTED),
                     flierprops=dict(marker='o', color=C_MUTED, markersize=3))
    for patch, color in zip(bp['boxes'], src_colors):
        patch.set_facecolor(color + "88")
        patch.set_edgecolor(color)
    ax3.axhline(y=SIGNIFICANCE, color=C_FAIL, ls='--', lw=1.5, label=f"α={SIGNIFICANCE}")
    ax3.axhline(y=0.5, color=C_GRID, ls=':', lw=1)
    ax3.set_xticks(range(1, len(src_names)+1))
    ax3.set_xticklabels([s.replace("_","\n") for s in src_names], fontsize=7, color=C_MUTED)
    ax3.set_ylabel("p-value", fontsize=8)
    ax3.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

    # ── Panel 4: Shannon & Min-entropy ───────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax_style(ax4, "④ Shannon & Min-Entropy (bits/bit)")
    x_pos   = np.arange(len(src_names))
    width   = 0.35
    shannon = [entropy_metrics[s]["shannon"] for s in src_names]
    minH    = [entropy_metrics[s]["min_entropy"] for s in src_names]
    # Normalize min-entropy to bits/bit (max=8 for byte-level)
    minH_norm = [m / 8 for m in minH]
    b1 = ax4.bar(x_pos - width/2, shannon, width, color=[c+"CC" for c in src_colors],
                 edgecolor=src_colors, label="Shannon H (bits/bit)")
    b2 = ax4.bar(x_pos + width/2, minH_norm, width,
                 color=[c+"55" for c in src_colors], edgecolor=src_colors,
                 hatch='//', label="Min-Entropy H∞/8")
    ax4.axhline(y=1.0, color=C_PASS, ls='--', lw=1.5, label="Perfect = 1.0")
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([s.replace("_","\n") for s in src_names], fontsize=7, color=C_MUTED)
    ax4.set_ylim(0, 1.15)
    ax4.set_ylabel("Entropy (normalized)", fontsize=8)
    ax4.legend(fontsize=6.5, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

    # ── Panel 5: Compression ratio ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax_style(ax5, "⑤ Compressibility (gzip ratio, 1.0 = incompressible)")
    comp = [entropy_metrics[s]["compression"] for s in src_names]
    bars5 = ax5.bar(range(len(src_names)), comp, color=src_colors,
                    edgecolor=C_GRID, width=0.6)
    ax5.axhline(y=1.0, color=C_PASS, ls='--', lw=1.5, label="Ideal random")
    ax5.set_xticks(range(len(src_names)))
    ax5.set_xticklabels([s.replace("_","\n") for s in src_names], fontsize=7, color=C_MUTED)
    ax5.set_ylabel("gzip output / input", fontsize=8)
    ax5.set_ylim(0.5, 1.15)
    ax5.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)
    for bar, v in zip(bars5, comp):
        ax5.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.3f}",
                 ha='center', color=C_TEXT, fontsize=7.5, fontweight='bold')

    # ── Panel 6: Autocorrelation ──────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    ax_style(ax6, "⑥ Autocorrelation (lags 1–100)  — ideal = 0 for all lags")
    lags = range(1, 101)
    for src in src_names:
        ac = entropy_metrics[src]["autocorr"]
        ax6.plot(lags, ac, color=SOURCES[src]["color"], lw=1.2,
                 alpha=0.85, label=SOURCES[src]["label"])
    ax6.axhline(y=0, color='white', lw=0.8, ls='--', alpha=0.5)
    ci = 1.96 / math.sqrt(N_BITS)
    ax6.fill_between(lags, -ci, ci, color='white', alpha=0.05, label=f"95% CI (±{ci:.5f})")
    ax6.set_xlabel("Lag", fontsize=8)
    ax6.set_ylabel("Autocorrelation", fontsize=8)
    ax6.legend(fontsize=7, facecolor=C_PANEL, edgecolor=C_GRID,
               labelcolor=C_TEXT, loc='upper right', ncol=2)

    # ── Panel 7: Bit distribution (first 10000 bits visualized) ─────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax_style(ax7, "⑦ Bit Density (sliding window, 100-bit blocks)")
    n_window = 10000
    block_w  = 100
    n_blocks = n_window // block_w
    x_w      = range(n_blocks)
    for src in src_names:
        blocks = all_bits[src][:n_window].reshape(n_blocks, block_w)
        density = blocks.mean(axis=1)
        ax7.plot(x_w, density, color=SOURCES[src]["color"],
                 lw=1.0, alpha=0.7)
    ax7.axhline(y=0.5, color='white', lw=1, ls='--', alpha=0.6, label="Expected 0.5")
    ax7.set_xlabel("Block index (×100 bits)", fontsize=8)
    ax7.set_ylabel("Bit density (proportion of 1s)", fontsize=8)
    ax7.set_ylim(0.2, 0.8)
    ax7.legend(fontsize=6, facecolor=C_PANEL, edgecolor=C_GRID, labelcolor=C_TEXT)

    # ── Panel 8: Scatter — p-value uniformity check ──────────────────────────
    ax8 = fig.add_subplot(gs[3, :])
    ax_style(ax8, "⑧ p-value per Test per Source — Ideal: all points above α=0.01 line")
    x_jitter = np.linspace(-0.15, 0.15, len(src_names))
    for j, src in enumerate(src_names):
        xs = np.arange(len(TEST_NAMES)) + x_jitter[j]
        ys = [r.p_value for r in all_results[src]]
        ax8.scatter(xs, ys, color=SOURCES[src]["color"],
                    marker=SOURCES[src]["marker"],
                    s=45, alpha=0.85, label=SOURCES[src]["label"],
                    edgecolors='none', zorder=3)
    ax8.axhline(y=SIGNIFICANCE, color=C_FAIL, ls='--', lw=2, label=f"α = {SIGNIFICANCE}")
    ax8.axhspan(0, SIGNIFICANCE, color=C_FAIL, alpha=0.07)
    ax8.set_xticks(range(len(TEST_NAMES)))
    ax8.set_xticklabels([f"T{i+1:02d}\n{TEST_NAMES[i].split(' ',1)[1][:14]}"
                         for i in range(len(TEST_NAMES))], fontsize=6.5, color=C_MUTED)
    ax8.set_ylabel("p-value", fontsize=8)
    ax8.set_ylim(-0.02, 1.05)
    ax8.legend(fontsize=7.5, facecolor=C_PANEL, edgecolor=C_GRID,
               labelcolor=C_TEXT, loc='upper right', ncol=3)

    out = "/mnt/user-data/outputs/true_entropy_nist_report.png"
    plt.savefig(out, dpi=145, bbox_inches='tight', facecolor=C_BG)
    plt.close()
    return out

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  TRUE ENTROPY TESTING — NIST SP 800-22 Statistical Test Suite")
    print("  Sources: QRNG_sim | TRNG (/dev/urandom) | MT19937 | LCG | Xorshift64")
    print("=" * 72)

    # ── Generate data ─────────────────────────────────────────────────────────
    print(f"\n[DATA]  Generating {N_BITS:,} bits per source …")
    all_bits = {}
    for name, gen_fn in GENERATORS.items():
        t0 = time.time()
        bits = gen_fn(N_BITS)
        dt   = time.time() - t0
        all_bits[name] = bits
        bias = abs(np.mean(bits) - 0.5)
        print(f"  {name:<14} {N_BITS:>8,} bits  |  bias={bias:.5f}  |  {dt:.2f}s")

    # ── Entropy metrics ───────────────────────────────────────────────────────
    print("\n[ENTROPY METRICS]")
    entropy_metrics = {}
    header = f"  {'Source':<14}  {'Shannon':>9}  {'Min-H∞':>9}  {'Compress':>9}  {'AutoCorr[1]':>12}"
    print(header)
    print("  " + "─" * 60)
    for name, bits in all_bits.items():
        sh   = shannon_entropy(bits)
        mh   = min_entropy(bits)
        comp = compression_ratio(bits)
        ac   = autocorrelation(bits, max_lag=100)
        entropy_metrics[name] = {
            "shannon":     sh,
            "min_entropy": mh,
            "compression": comp,
            "autocorr":    ac,
        }
        print(f"  {name:<14}  {sh:>9.6f}  {mh:>9.4f}  {comp:>9.4f}  {ac[0]:>12.6f}")

    # ── Run NIST tests ────────────────────────────────────────────────────────
    print(f"\n[NIST SP 800-22]  Running 15 tests × {len(GENERATORS)} sources …")
    all_results = {}
    for name, bits in all_bits.items():
        print(f"\n  ── {name} ──")
        results = run_all_tests(bits, name)
        all_results[name] = results
        passed = sum(1 for r in results if r.passed)
        for r in results:
            status = "✓ PASS" if r.passed else "✗ FAIL"
            print(f"    {r.name:<35} p={r.p_value:.5f}  [{status}]")
        print(f"  → {passed}/{len(results)} tests passed")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY")
    print("  " + "─" * 68)
    print(f"  {'Source':<14}  {'Pass/15':>7}  {'Pass%':>6}  {'Shannon':>9}  {'Min-H∞':>8}  {'Compress':>9}")
    print("  " + "─" * 68)
    for name in src_names:
        passed = sum(1 for r in all_results[name] if r.passed)
        pct    = passed / len(TEST_NAMES) * 100
        sh     = entropy_metrics[name]["shannon"]
        mh     = entropy_metrics[name]["min_entropy"]
        comp   = entropy_metrics[name]["compression"]
        mark   = "★" if passed == 15 else ("⚠" if passed >= 12 else "✗")
        print(f"  {name:<14}  {passed:>4}/15  {pct:>5.0f}%  {sh:>9.6f}  {mh:>8.4f}  {comp:>9.4f}  {mark}")
    print("=" * 72)

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n[VERDICT]  Does 'quantumness' provide a superior entropy profile?")
    qrng_pass  = sum(1 for r in all_results["QRNG_sim"]  if r.passed)
    trng_pass  = sum(1 for r in all_results["TRNG"]       if r.passed)
    mt_pass    = sum(1 for r in all_results["MT19937"]    if r.passed)
    lcg_pass   = sum(1 for r in all_results["LCG"]        if r.passed)
    xs_pass    = sum(1 for r in all_results["Xorshift64"] if r.passed)
    print(f"  QRNG_sim  passed {qrng_pass}/15  — quantum physical model")
    print(f"  TRNG      passed {trng_pass}/15  — /dev/urandom (CSPRNG + hardware entropy)")
    print(f"  MT19937   passed {mt_pass}/15  — best-in-class PRNG")
    print(f"  LCG       passed {lcg_pass}/15  — weak classical PRNG")
    print(f"  Xorshift  passed {xs_pass}/15  — fast modern PRNG")
    print()
    if qrng_pass >= trng_pass >= mt_pass:
        print("  ► QRNG and TRNG both achieve top-tier scores.")
        print("    Statistical tests cannot distinguish quantum from good classical RNG.")
        print("    The advantage of QRNG lies in UNPREDICTABILITY (non-algorithmic),")
        print("    not in better statistics. A good CSPRNG like /dev/urandom is")
        print("    statistically indistinguishable but computationally predictable")
        print("    if the seed/state is compromised.")
    if lcg_pass < mt_pass:
        print(f"\n  ► LCG ({lcg_pass}/15) clearly inferior — linear structure exposed.")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\n[PLOT]  Generating 8-panel NIST report figure …")
    out = generate_report(all_results, all_bits, entropy_metrics)
    print(f"  Saved → {out}")
    print("=" * 72)

    return all_results, entropy_metrics, all_bits

src_names = list(SOURCES.keys())

if __name__ == "__main__":
    main()
