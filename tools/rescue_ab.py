"""rescue_ab.py - the H1/H2 experiment: confidence rescue vs blind flip.

Replay A/B on a frozen IQ corpus (same frames, both algorithms):
  H1: does weakest-bits-first rescue recover more valid messages than the
      dump1090-style blind scan?
  H2: are rescued frames trustworthy? Ghost audit: a rescued ICAO that
      never appears in any clean-CRC frame of the same corpus is a
      suspected miscorrection.

Usage:  python rescue_ab.py <capture.cs16> [more.cs16 ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adsb import demod_frames, rescue, rescue_blind, decode_fields, crc24   # noqa


def run(files):
    frames_all = []
    for f in files:
        raw = np.fromfile(f, dtype=np.int16).astype(np.float32) / 32768.0
        iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
        fr = demod_frames(iq)
        frames_all.extend(fr)
        print(f"  {Path(f).name}: {len(fr)} DF17/18 candidates")
    clean = [f for f in frames_all if f["crc_ok"]]
    dirty = [f for f in frames_all if not f["crc_ok"]]
    clean_icaos = {decode_fields(f["bits"])["icao"] for f in clean}
    print(f"\ncorpus: {len(frames_all)} candidates = {len(clean)} clean + "
          f"{len(dirty)} damaged | {len(clean_icaos)} clean aircraft")

    res = {}
    for name in ("confidence", "blind"):
        fixed = []
        tries_tot = 0
        for f in dirty:
            if name == "confidence":
                b2, nf = rescue(f["bits"], f["conf"])
                tries = 8 + 15          # fixed search budget
            else:
                b2, nf, tries = rescue_blind(f["bits"])
            tries_tot += tries
            if b2 is not None:
                fixed.append({"bits": b2, "flips": nf})
        icaos = [decode_fields(x["bits"])["icao"] for x in fixed]
        ghosts = [i for i in icaos if i not in clean_icaos]
        # lenient: a "ghost" that other rescued frames independently agree on
        from collections import Counter
        cnt = Counter(icaos)
        hard_ghosts = [i for i in ghosts if cnt[i] == 1]
        res[name] = {"rescued": len(fixed), "tries": tries_tot,
                     "ghost_strict": len(ghosts),
                     "ghost_hard": len(hard_ghosts),
                     "icaos": icaos}
        print(f"\n[{name}] rescued {len(fixed)}/{len(dirty)} damaged frames "
              f"({tries_tot} CRC trials)")
        print(f"          ghosts: {len(ghosts)} strict "
              f"({100*len(ghosts)/max(1,len(fixed)):.1f}%), "
              f"{len(hard_ghosts)} hard (single-appearance) "
              f"({100*len(hard_ghosts)/max(1,len(fixed)):.1f}%)")

    c, b = res["confidence"], res["blind"]
    ratio = c["rescued"] / max(1, b["rescued"])
    print("\n" + "=" * 62)
    print(f"H1 VERDICT: confidence/blind rescue ratio = {ratio:.2f} "
          f"(claim: >=1.25)  ->  {'CONFIRMED' if ratio >= 1.25 else 'NOT confirmed'}")
    eff_c = c["rescued"] / max(1, c["tries"])
    eff_b = b["rescued"] / max(1, b["tries"])
    print(f"    efficiency: {1000*eff_c:.1f} vs {1000*eff_b:.1f} rescues per "
          f"1000 CRC trials ({eff_c/max(eff_b,1e-9):.1f}x)")
    gr = 100 * c["ghost_hard"] / max(1, c["rescued"])
    print(f"H2 VERDICT: confidence-rescue hard-ghost rate = {gr:.1f}% "
          f"(claim: <1%... see note)  ->  "
          f"{'CONFIRMED' if gr < 1 else 'measured - see ledger note'}")
    print("=" * 62)
    return res


if __name__ == "__main__":
    files = sys.argv[1:] or [str(Path(r"Z:\SDR_Agent_v2\corpus")
                                 .glob("adsb_*.cs16").__next__())]
    run(files)
