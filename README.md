# aeroTuna ✈️📡

**Adaptive ADS-B decoding — the TV Tuna method pointed at 1090 MHz.**

Born 2026-07-17 from the [Software-TV-Tuner](https://github.com/Felbs/Software-TV-Tuner)
lineage (TV Tuna → [Radio Tuna](https://github.com/Felbs/gr-radiotuna) →
[wxTuna](https://github.com/Felbs/wxTuna) → aeroTuna). Same thesis, new sky:
every decoder secretly knows how well it's doing — close the loop on it.

## The idea
Every Mode S message carries a 24-bit CRC. Stock decoders (the dump1090
family) rescue corrupted messages by **blind bit-flipping** against that
CRC. aeroTuna demodulates with a **per-bit confidence plane** (the energy
margin between the two PPM chips — the SOVA idea from our ATSC decoder),
so rescue flips the *weakest* bits first. Smarter search, more recovered
messages from weak, distant aircraft.

**First-light numbers (indoor antenna, 20 s):** 30 CRC-valid messages raw,
**+81 recovered by confidence-guided rescue** — 2.7× more traffic than the
raw demod alone. 17 aircraft decoded, callsigns and all.

## Tools
| Command | What it does |
|---|---|
| `python tools/adsb.py selftest` | No radio needed: CRC-24 against published Mode S test vectors, field decode, synthetic-IQ roundtrip, and a marginal-bit rescue proof. |
| `python tools/adsb.py capture --secs 20` | Live 1090 MHz: demod → CRC gate → confidence rescue → plane table (ICAO, callsign, altitude, speed). |
| `python tools/adsb.py shootout` | Antenna A/B/C compared by decoded-message count — pick your 1090 MHz antenna empirically, not by folklore. |

Hot loops are numba-jitted; a 20 s capture analyzes in ~3 s.

## Status (early)
- ✅ Demod + CRC + rescue validated (selftest) and proven live (17 aircraft first capture)
- ✅ Antenna shootout working — measure, don't assume (our indoor rabbit ears beat two bigger antennas at 1090)
- ⏳ Next: CPR position decode (lat/lon), live map, rescue-vs-dump1090 A/B harness, miscorrection audit on rescued frames

## Hardware
Any SoapySDR-supported SDR (reference: SDRplay RSPdx) + any antenna — the
shootout tells you which one. 1090 MHz loves short coax and line of sight.

## License
MIT
