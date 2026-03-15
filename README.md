# True Entropy Testing

> Can a quantum random number generator actually beat a normal computer's random number generator? This project finds out — using real statistical tests.

---

## What This Project Does

This project compares five different sources of random numbers and runs them through the official **NIST SP 800-22** test suite — the same tests used by governments and security labs to certify randomness.

The five sources tested are:

| Source | Type | Description |
|--------|------|-------------|
| **QRNG_sim** | Quantum (simulated) | Models photon shot noise + vacuum fluctuations |
| **TRNG** | Hardware | Reads from `/dev/urandom` on Linux |
| **MT19937** | Classical PRNG | Python's built-in Mersenne Twister |
| **LCG** | Weak PRNG | Old-school linear formula, used to show what bad looks like |
| **Xorshift64** | Modern PRNG | Fast shift-register generator |

---

## What We Found

| Source | Tests Passed | gzip ratio | Verdict |
|--------|-------------|------------|---------|
| QRNG_sim | **15 / 15** | 1.0006 | ✅ Perfect |
| TRNG (`/dev/urandom`) | **15 / 15** | 1.0006 | ✅ Perfect |
| MT19937 | **15 / 15** | 1.0006 | ✅ Perfect |
| LCG | **9 / 15** | **0.14** | ❌ Fails badly |
| Xorshift64 | **15 / 15** | 1.0006 | ✅ Perfect |

**The big finding:** Statistical tests alone cannot tell the difference between quantum random and a good classical generator. They look identical. The difference only shows up when you ask: what happens if someone steals the internal state?

- A quantum device: still unpredictable — the next event has not happened yet.
- A math formula: fully reconstructible — an attacker can replay every number ever generated.

LCG is the cautionary tale. Its Shannon entropy looks perfect (1.0 bits/bit) yet it fails 6 tests and compresses down to 14% of its original size. The linear pattern is so obvious a compressor can model it completely.

---

## How to Run It

**Requirements:** Python 3.9+, Linux (for `/dev/urandom`)

```bash
pip install numpy matplotlib scipy
python true_entropy_testing.py
```

That's it. It will generate all the data, run the 15 tests across all 5 sources, print a full results table, and save the figure.

Expected runtime: around 20–30 seconds.

---

## The 15 NIST SP 800-22 Tests

The script implements every test from the official NIST standard:

1. Monobit Frequency — are there roughly equal 0s and 1s?
2. Block Frequency — does balance hold in smaller chunks too?
3. Runs — do runs of 0s and 1s switch at the right rate?
4. Longest Run of Ones — are there suspiciously long streaks?
5. Binary Matrix Rank — is there hidden linear structure?
6. DFT / Spectral — are there repeating patterns in frequency space?
7. Non-overlapping Template — does a specific bit pattern appear too often?
8. Overlapping Template — same, but allowing overlaps
9. Universal (Maurer's) — how compressible is the sequence?
10. Linear Complexity — how long does a formula need to be to copy the sequence?
11. Serial — do pairs and triples of bits appear evenly?
12. Approximate Entropy — how predictable is each bit given the previous ones?
13. Cumulative Sums — does the running total stay near zero?
14. Random Excursions — does a random walk behave normally?
15. Random Excursions Variant — same, checking more states

---

## The Output Figure

The script saves `true_entropy_nist_report.png` with 8 panels:

- **Heatmap** — pass/fail for every test × every source at a glance
- **Pass rate bars** — who passed how many tests
- **Box plots** — spread of p-values per source
- **Entropy bars** — Shannon entropy and min-entropy comparison
- **Compressibility** — gzip ratio (lower = more structure = worse)
- **Autocorrelation** — does knowing one bit help predict the next?
- **Bit density** — does the 0/1 balance stay stable over time?
- **p-value scatter** — every single test result in one view

---

## What the Research Notes Cover

The `research_notes.txt` and `research_notes.docx` files cover the background in plain language:

- What **ISO/IEC 23837** actually requires (the standard for certifying quantum RNG hardware)
- The three main physical methods used in real QRNG devices: beam splitter, vacuum fluctuations, and radioactive decay
- Whether `/dev/random` on Linux counts as a true hardware source or a fake formula (the answer is: both, kind of)
- A simple comparison table showing which source is right for which use

---

## Key Concepts (Plain Version)

**QRNG** — A device that uses real quantum physics events (like a photon hitting a mirror) to generate random numbers. The outcome is not predictable even in theory.

**TRNG** — True Random Number Generator. Gets its randomness from physical hardware events like mouse movements, disk timing, or a thermal noise chip.

**PRNG** — Pseudo-Random Number Generator. Uses a math formula. Looks random but is actually deterministic — if you know the starting seed, you can reproduce every number.

**NIST SP 800-22** — The US government's official test suite for randomness. Passing all 15 tests is required for cryptographic certification.

**Entropy** — A measure of unpredictability. 1.0 bits/bit means perfectly unpredictable. Below that means some pattern exists.

---

## License

MIT — free to use and modify.
