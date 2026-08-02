"""referee_pymodes.py - independent field-decode referee.

Runs our demod over a frozen corpus, hands every CRC-native frame's hex
to pyModeS (the community-standard ADS-B library, pinned <3 for the
function-per-field API) and compares field by field: ICAO, callsign,
altitude, ground speed, track, vertical rate. An external decoder
agreeing bit-for-bit is the referee's stamp our own selftest cannot
give itself.

First run (2026-08-02, 240 s corpus): 279 frames, ZERO mismatches on
every field.

  pip install "pyModeS<3"
  python referee_pymodes.py [corpus.cs16 ...]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adsb import FS, demod_frames, decode_fields, bits_to_hex  # noqa: E402

DEFAULT = [Path(r"Z:\SDR_Agent_v2\corpus\adsb_20260717_195934.cs16")]


def run(files):
    import pyModeS as pms
    frames = []
    for f in files:
        raw = np.fromfile(f, np.int16)
        iq = ((raw[0::2].astype(np.float32)
               + 1j * raw[1::2].astype(np.float32)) / 32768.0
              ).astype(np.complex64)
        step = int(2.0 * FS)
        for a in range(0, len(iq) - step, step):
            frames += [fr for fr in demod_frames(iq[a:a + step + 300])
                       if fr["kind"] == "es" and fr["crc_ok"]]
    mismatch = {k: 0 for k in ("icao", "callsign", "alt",
                               "speed", "track", "vr")}
    checked = dict(mismatch)
    examples = []
    for fr in frames:
        h = bits_to_hex(fr["bits"])
        d = decode_fields(fr["bits"])
        if pms.crc(h) != 0:
            continue
        checked["icao"] += 1
        ic = pms.icao(h)
        if ic and ic.upper() != d["icao"]:
            mismatch["icao"] += 1
        tc = pms.adsb.typecode(h)
        if tc and 1 <= tc <= 4:
            checked["callsign"] += 1
            cs = pms.adsb.callsign(h).replace("_", "").strip()
            if cs != d.get("callsign", ""):
                mismatch["callsign"] += 1
                examples.append(("callsign", cs, d.get("callsign")))
        elif tc and 9 <= tc <= 18:
            ref = pms.adsb.altitude(h)
            if ref is not None and "alt_ft" in d:
                checked["alt"] += 1
                if abs(ref - d["alt_ft"]) > 0.1:
                    mismatch["alt"] += 1
                    examples.append(("alt", ref, d["alt_ft"]))
        elif tc == 19:
            v = pms.adsb.velocity(h)
            for key, ours, idx, tol in (("speed", "speed_kt", 0, 1),
                                        ("track", "track_deg", 1, 1),
                                        ("vr", "vr_fpm", 2, 1)):
                if v and v[idx] is not None and ours in d:
                    checked[key] += 1
                    diff = abs(v[idx] - d[ours])
                    if key == "track":
                        diff = min(diff % 360, 360 - diff % 360)
                    if diff > tol:
                        mismatch[key] += 1
                        examples.append((key, v[idx], d[ours]))
    print(f"{sum(1 for _ in frames)} native frames refereed by pyModeS:")
    print(f"{'field':<12}{'checked':>8}{'mismatch':>9}")
    ok = True
    for k in checked:
        print(f"{k:<12}{checked[k]:>8}{mismatch[k]:>9}")
        ok &= mismatch[k] == 0
    for e in examples[:10]:
        print("  mismatch:", e)
    print("REFEREE:", "CLEAN - external decoder agrees on every field"
          if ok else "DISAGREEMENTS FOUND - investigate above")
    return 0 if ok else 1


if __name__ == "__main__":
    files = [Path(p) for p in sys.argv[1:]] or DEFAULT
    sys.exit(run([f for f in files if f.is_file()]))
