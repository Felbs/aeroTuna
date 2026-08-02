"""miscorrect_audit.py - do rescued frames LIE? The H2 follow-up.

A rescued frame passed CRC after flipping bits - but a 24-bit CRC over
112 bits leaves room for plausible-but-wrong repairs. This audit
cross-examines every rescued frame against the same aircraft's
CRC-NATIVE truth (frames that passed with zero flips):

  callsign  - must equal the native callsign exactly
  altitude  - within 1500 ft of the native altitude nearest in time
  speed     - within 60 kt              (nearest native within 60 s)
  track     - within 25 degrees
  position  - CPR local decode within 15 nm of nearest native fix

Frames from aircraft never heard natively are 'unverifiable' and
counted separately (they are the ghost-aircraft risk H2 bounded).

The same damaged frames are also repaired with the dump1090-style
BLIND flip (rescue_blind) and audited identically - the A/B that says
whether the confidence plane miscorrects less, not just rescues more.

  python miscorrect_audit.py [corpus.cs16 ...]   # default: repo corpus
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adsb import (FS, demod_frames, rescue, rescue_blind, decode_fields,
                  cpr_local, cpr_global, _bf)  # noqa: E402

DEFAULT = [Path(r"Z:\SDR_Agent_v2\corpus\adsb_20260717_195934.cs16")]


def collect(files):
    """Demod whole corpus -> (native, damaged) ES frames with times."""
    native, damaged = [], []
    t_base = 0.0
    for f in files:
        raw = np.fromfile(f, np.int16)
        iq = ((raw[0::2].astype(np.float32)
               + 1j * raw[1::2].astype(np.float32)) / 32768.0
              ).astype(np.complex64)
        step = int(2.0 * FS)
        for a in range(0, len(iq) - step, step):
            for fr in demod_frames(iq[a:a + step + 300]):
                if fr["kind"] != "es":
                    continue
                fr["t"] = t_base + (a + fr["start"]) / FS
                (native if fr["crc_ok"] else damaged).append(fr)
        t_base += len(iq) / FS
        print(f"  {f.name}: cumulative {len(native)} native, "
              f"{len(damaged)} damaged")
    return native, damaged


def truth_tables(native):
    """Per-aircraft state from CRC-native frames only."""
    T = {}
    for fr in sorted(native, key=lambda x: x["t"]):
        d = decode_fields(fr["bits"])
        a = T.setdefault(d["icao"], {"cs": set(), "alt": [], "spd": [],
                                     "trk": [], "pos": [], "cpr": {}})
        t = fr["t"]
        if "callsign" in d:
            a["cs"].add(d["callsign"])
        if "alt_ft" in d:
            a["alt"].append((t, d["alt_ft"]))
        if "speed_kt" in d:
            a["spd"].append((t, d["speed_kt"]))
        if "track_deg" in d:
            a["trk"].append((t, d["track_deg"]))
        if "lat_cpr" in d:
            a["cpr"][d["cpr_odd"]] = (d["lat_cpr"], d["lon_cpr"], t)
            if 0 in a["cpr"] and 1 in a["cpr"]:
                e, o = a["cpr"][0], a["cpr"][1]
                if abs(e[2] - o[2]) <= 10.0:
                    g = cpr_global(e[0], e[1], o[0], o[1], o[2] > e[2])
                    if g:
                        a["pos"].append((t, g[0], g[1]))
    return T


def nearest(series, t, window=60.0):
    best = None
    bd = window
    for ts, v in series:
        d = abs(ts - t)
        if d < bd:
            bd = d
            best = v
    return best


def audit_one(bits, t, truth):
    """Field-check one repaired frame against native truth.
    Returns (n_checked, mismatches:list[str]) - unverifiable = (0, [])."""
    d = decode_fields(bits)
    a = truth.get(d["icao"])
    if a is None:
        return 0, ["ghost-icao"]
    checked = 0
    bad = []
    if "callsign" in d and a["cs"]:
        checked += 1
        if d["callsign"] not in a["cs"]:
            bad.append(f"callsign {d['callsign']} vs {sorted(a['cs'])}")
    if "alt_ft" in d:
        ref = nearest(a["alt"], t)
        if ref is not None:
            checked += 1
            if abs(d["alt_ft"] - ref) > 1500:
                bad.append(f"alt {d['alt_ft']} vs {ref}")
    if "speed_kt" in d:
        ref = nearest(a["spd"], t)
        if ref is not None:
            checked += 1
            if abs(d["speed_kt"] - ref) > 60:
                bad.append(f"speed {d['speed_kt']} vs {ref}")
    if "track_deg" in d:
        ref = nearest(a["trk"], t)
        if ref is not None:
            checked += 1
            diff = abs(d["track_deg"] - ref) % 360
            if min(diff, 360 - diff) > 25:
                bad.append(f"track {d['track_deg']:.0f} vs {ref:.0f}")
    if "lat_cpr" in d and a["pos"]:
        ref = min(a["pos"], key=lambda p: abs(p[0] - t))
        if abs(ref[0] - t) < 60:
            checked += 1
            lat, lon = cpr_local(d["lat_cpr"], d["lon_cpr"], d["cpr_odd"],
                                 ref[1], ref[2])
            import math
            dnm = 60 * math.hypot(lat - ref[1], (lon - ref[2])
                                  * math.cos(math.radians(lat)))
            if dnm > 15:
                bad.append(f"pos {dnm:.0f} nm off")
    return checked, bad


def run(files):
    print(f"[audit] corpus: {[f.name for f in files]}")
    native, damaged = collect(files)
    truth = truth_tables(native)
    print(f"[audit] truth: {len(truth)} aircraft from {len(native)} "
          f"native frames; auditing repairs of {len(damaged)} damaged")
    results = {}
    for method, fixer in (("confidence", lambda f: rescue(f["bits"], f["conf"])),
                          ("blind", lambda f: rescue_blind(f["bits"])[:2])):
        n_fix = n_verif = n_checked = n_bad = n_ghost = 0
        examples = []
        for fr in damaged:
            b2, nf = fixer(fr)
            if b2 is None:
                continue
            n_fix += 1
            checked, bad = audit_one(b2, fr["t"], truth)
            if bad == ["ghost-icao"]:
                n_ghost += 1
                continue
            if checked:
                n_verif += 1
                n_checked += checked
                if bad:
                    n_bad += 1
                    if len(examples) < 6:
                        examples.append(bad)
        results[method] = (n_fix, n_verif, n_checked, n_bad, n_ghost,
                          examples)
    print()
    print("=" * 66)
    print(f"{'method':<12}{'repaired':>9}{'verifiable':>11}{'field-chk':>10}"
          f"{'BAD':>6}{'bad %':>8}{'ghost-icao':>11}")
    for m, (nf, nv, nc, nb, ng, ex) in results.items():
        pct = 100.0 * nb / nv if nv else 0.0
        print(f"{m:<12}{nf:>9}{nv:>11}{nc:>10}{nb:>6}{pct:>7.1f}%{ng:>11}")
    print("=" * 66)
    for m, (nf, nv, nc, nb, ng, ex) in results.items():
        if ex:
            print(f"{m} mismatch examples:")
            for e in ex:
                print(f"    {e}")
    return results


if __name__ == "__main__":
    files = [Path(p) for p in sys.argv[1:]] or DEFAULT
    run([f for f in files if f.is_file()])
