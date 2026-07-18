"""adsb.py - aeroTuna campaign 1: ADS-B / Mode S with a confidence plane.

The TV Tuna method pointed at 1090 MHz: every Mode S message carries a
24-bit CRC (the truth dial), planes never stop broadcasting (a 24/7 bench),
and the stock decoders rescue corrupted messages by BLIND bit-flipping
against the CRC. We demodulate with per-bit CONFIDENCE (|chip energy
difference|, the SOVA idea from our ATSC work) so rescue can flip the
weakest bits first - measurably smarter on weak, distant aircraft.

Pipeline: IQ @ 2 MS/s -> magnitude -> preamble correlate -> PPM bits +
confidence -> CRC-24 gate -> decode (ICAO, callsign, altitude, velocity).
All hot loops numba-jitted from day one (see wxTuna's 31 GB lesson).

Modes:
  selftest   - CRC known-vectors + synthetic-IQ roundtrip (no SDR)
  capture    - N seconds of live 1090 MHz -> decode -> plane table
  shootout   - antenna A/B/C compared by DECODED MESSAGE COUNT (the dial)

Examples:
  python adsb.py selftest
  python adsb.py capture --secs 20 --antenna "Antenna B"
  python adsb.py shootout --secs 15
"""
import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

HERE = Path(__file__).resolve().parent
LAB = HERE.parent / "lab"
LAB.mkdir(exist_ok=True)

FS = 2_000_000.0          # 2 MS/s -> 1 sample per 0.5 us chip
FREQ = 1090e6


def _ensure_sdr_dll_path():
    """Bare radioconda python.exe can't load SoapySDR driver DLLs without
    the conda Library\\bin + SDRplay API dirs on the search path."""
    if os.name != "nt":
        return
    root = Path(sys.executable).resolve().parent
    for p in (root / "Library" / "bin",
              Path(r"C:\Program Files\SDRplay\API\x64"),
              Path(r"C:\Program Files\SDRplay\API")):
        if p.is_dir():
            os.environ["PATH"] = str(p) + os.pathsep + os.environ["PATH"]
            try:
                os.add_dll_directory(str(p))
            except Exception:
                pass


_ensure_sdr_dll_path()


# ==========================================================================
# Mode S CRC-24  (generator 0x1FFF409 as a 25-bit polynomial)
# ==========================================================================
def crc24(bits):
    """Remainder of the full 56/112-bit message; 0 == valid for DF17/18."""
    reg = 0
    for b in bits:
        reg = (reg << 1) | int(b)
        if reg & (1 << 24):
            reg ^= 0x1FFF409
    return reg & 0xFFFFFF


def hex_to_bits(h):
    v = int(h, 16)
    n = len(h) * 4
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], np.uint8)


def bits_to_hex(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return f"{v:0{len(bits)//4}X}"


# ==========================================================================
# PPM demod: magnitude -> preamble scan -> bits + per-bit confidence
# ==========================================================================
# Preamble chips (0.5 us each): pulses at 0, 1.0, 3.5, 4.5 us
_PRE_HI = np.array([0, 2, 7, 9], np.int64)
_PRE_LO = np.array([1, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15], np.int64)


def _scan_impl(mag, pre_hi, pre_lo, thresh_ratio, floor):
    """Return (starts, bits, conf) for every plausible 112-bit frame.
    A hit requires the WEAKEST preamble pulse to beat both the local gap
    average and an absolute floor - kills noise false-fires that would
    skip past real frames."""
    N = mag.shape[0]
    max_msgs = 4096
    starts = np.empty(max_msgs, np.int64)
    bits = np.empty((max_msgs, 112), np.uint8)
    conf = np.empty((max_msgs, 112), np.float32)
    nmsg = 0
    i = 0
    while i < N - 16 - 224 and nmsg < max_msgs:
        hi = mag[i + pre_hi[0]]
        hi_min = hi
        for k in range(1, 4):
            v = mag[i + pre_hi[k]]
            hi += v
            if v < hi_min:
                hi_min = v
        hi *= 0.25
        lo = 0.0
        for k in range(12):
            lo += mag[i + pre_lo[k]]
        lo /= 12.0
        if hi_min > thresh_ratio * lo and hi_min > floor:
            base = i + 16
            ok = True
            score = 0.0
            for b in range(112):
                c0 = mag[base + 2 * b]
                c1 = mag[base + 2 * b + 1]
                bits[nmsg, b] = 1 if c0 > c1 else 0
                d = c0 - c1
                conf[nmsg, b] = d if d >= 0 else -d
                score += conf[nmsg, b]
            if ok and score > 0.0:
                starts[nmsg] = i
                nmsg += 1
                i += 240            # skip past this frame
                continue
        i += 1
    return starts[:nmsg], bits[:nmsg], conf[:nmsg]


if _HAVE_NUMBA:
    _scan = njit(cache=True)(_scan_impl)
else:
    _scan = _scan_impl


def demod_frames(iq, thresh_ratio=1.5):
    """IQ (complex64 @ 2 MS/s) -> list of dicts with bits/conf/CRC status."""
    mag = np.abs(iq).astype(np.float32)
    floor = 3.5 * float(np.median(mag))
    starts, bits, conf = _scan(mag, _PRE_HI, _PRE_LO, thresh_ratio, floor)
    out = []
    for k in range(len(starts)):
        b = bits[k]
        df = (int(b[0]) << 4) | (int(b[1]) << 3) | (int(b[2]) << 2) \
            | (int(b[3]) << 1) | int(b[4])
        if df not in (17, 18):      # extended squitter only (v1)
            continue
        rem = crc24(b)
        out.append({"start": int(starts[k]), "df": df, "bits": b,
                    "conf": conf[k], "crc_ok": rem == 0, "rem": rem})
    return out


def rescue(bits, conf, max_flips=2):
    """Confidence-guided repair: try flipping the weakest 1-2 bits.
    (The smarter cousin of dump1090's blind single-bit scan.)"""
    order = np.argsort(conf)          # weakest first
    for i in range(min(8, len(order))):
        b2 = bits.copy()
        b2[order[i]] ^= 1
        if crc24(b2) == 0:
            return b2, 1
    if max_flips >= 2:
        for i in range(min(6, len(order))):
            for j in range(i + 1, min(6, len(order))):
                b2 = bits.copy()
                b2[order[i]] ^= 1
                b2[order[j]] ^= 1
                if crc24(b2) == 0:
                    return b2, 2
    return None, 0


# ==========================================================================
# ADS-B field decode (v1: ICAO, callsign, altitude, velocity)
# ==========================================================================
_CHARSET = "#ABCDEFGHIJKLMNOPQRSTUVWXYZ##### ###############0123456789######"


def _bf(bits, a, b):
    v = 0
    for i in range(a, b):
        v = (v << 1) | int(bits[i])
    return v


def decode_fields(bits):
    icao = f"{_bf(bits, 8, 32):06X}"
    tc = _bf(bits, 32, 37)
    info = {"icao": icao, "tc": tc}
    if 1 <= tc <= 4:                   # identification: callsign
        cs = ""
        for k in range(8):
            cs += _CHARSET[_bf(bits, 40 + 6 * k, 46 + 6 * k)]
        info["callsign"] = cs.replace("#", "").strip()
    elif 9 <= tc <= 18:                # airborne position: altitude (Q-bit)
        q = int(bits[47])
        if q:
            n = (_bf(bits, 40, 47) << 4) | _bf(bits, 48, 52)
            info["alt_ft"] = n * 25 - 1000
    elif tc == 19:                     # velocity, subtype 1/2
        st = _bf(bits, 37, 40)
        if st in (1, 2):
            vew = _bf(bits, 46, 56) - 1
            vns = _bf(bits, 57, 67) - 1
            if vew >= 0 and vns >= 0:
                info["speed_kt"] = int(round(math.hypot(vew, vns)))
    return info


# ==========================================================================
# SDR capture
# ==========================================================================
def open_sdr(antenna, gain_db=45, fs=FS):
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
    SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
    sdr = SoapySDR.Device("driver=sdrplay")
    sdr.setSampleRate(SOAPY_SDR_RX, 0, fs)
    sdr.setFrequency(SOAPY_SDR_RX, 0, FREQ)
    try:
        sdr.setAntenna(SOAPY_SDR_RX, 0, antenna)
    except Exception:
        pass
    try:
        sdr.setGainMode(SOAPY_SDR_RX, 0, False)
    except Exception:
        pass
    try:
        sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", max(20, 59 - gain_db))
        sdr.writeSetting("rfgain_sel", "0")   # max RF gain for 1 GHz
    except Exception:
        pass
    st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16)
    sdr.activateStream(st)
    return sdr, st


def capture_iq(secs, antenna, gain_db=45):
    sdr, st = open_sdr(antenna, gain_db)
    n_want = int(secs * FS)
    buf = np.empty(2 * 65536, np.int16)
    out = np.empty(2 * n_want, np.int16)
    got = 0
    while got < n_want:
        r = sdr.readStream(st, [buf], 65536, timeoutUs=1_000_000)
        if r.ret > 0:
            n = min(r.ret, n_want - got)
            out[2 * got: 2 * (got + n)] = buf[:2 * n]
            got += n
        elif r.ret < 0 and r.ret != -1:
            break
    sdr.deactivateStream(st)
    sdr.closeStream(st)
    iq = (out[0::2].astype(np.float32) + 1j * out[1::2].astype(np.float32)) / 32768.0
    return iq[:got].astype(np.complex64)


def analyze(iq, do_rescue=True):
    frames = demod_frames(iq)
    good = [f for f in frames if f["crc_ok"]]
    rescued = []
    if do_rescue:
        for f in frames:
            if not f["crc_ok"]:
                b2, nf = rescue(f["bits"], f["conf"])
                if b2 is not None:
                    rescued.append({**f, "bits": b2, "flips": nf})
    planes = {}
    for f in good + rescued:
        d = decode_fields(f["bits"])
        p = planes.setdefault(d["icao"], {"msgs": 0})
        p["msgs"] += 1
        for key in ("callsign", "alt_ft", "speed_kt"):
            if key in d:
                p[key] = d[key]
    return {"candidates": len(frames), "crc_ok": len(good),
            "rescued": len(rescued), "planes": planes}


# ==========================================================================
# commands
# ==========================================================================
def cmd_selftest(args):
    print("=" * 62)
    print("aeroTuna ADS-B self-test")
    print("=" * 62)
    ok = True
    # 1. CRC on published Mode S test vectors (mode-s.org examples)
    print("[1] CRC-24 known vectors")
    for h in ("8D4840D6202CC371C32CE0576098",
              "8D40621D58C382D690C8AC2863A7"):
        r = crc24(hex_to_bits(h))
        print(f"    {h[:14]}... remainder={r}  {'OK' if r == 0 else 'FAIL'}")
        ok &= (r == 0)
    # 2. field decode of the known vectors
    print("[2] field decode")
    d1 = decode_fields(hex_to_bits("8D4840D6202CC371C32CE0576098"))
    print(f"    callsign vector -> icao={d1['icao']} callsign={d1.get('callsign')}"
          f"  {'OK' if d1.get('callsign') == 'KLM1023' else 'FAIL'}")
    ok &= d1.get("callsign") == "KLM1023"
    d2 = decode_fields(hex_to_bits("8D40621D58C382D690C8AC2863A7"))
    print(f"    position vector -> icao={d2['icao']} alt={d2.get('alt_ft')} ft"
          f"  {'OK' if d2.get('alt_ft') == 38000 else 'FAIL'}")
    ok &= d2.get("alt_ft") == 38000
    # 3. synthetic IQ roundtrip (+ noise), incl. confidence-guided rescue
    print("[3] synthetic IQ roundtrip")
    rng = np.random.default_rng(1)
    msg = hex_to_bits("8D4840D6202CC371C32CE0576098")
    sig = np.zeros(4000, np.float32)
    pos = 1000
    for k, chip in enumerate([1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]):
        sig[pos + k] = chip
    for b in range(112):
        sig[pos + 16 + 2 * b + (0 if msg[b] else 1)] = 1.0
    iq = (sig * (0.5 + 0j)).astype(np.complex64)
    iq += (rng.normal(0, 0.03, len(sig)) + 1j * rng.normal(0, 0.03, len(sig))
           ).astype(np.complex64)
    res = analyze(iq, do_rescue=False)
    hit = res["crc_ok"] == 1
    print(f"    clean+noise: candidates={res['candidates']} crc_ok={res['crc_ok']}"
          f"  {'OK' if hit else 'FAIL'}")
    ok &= hit
    # make ONE bit marginal-and-wrong (nearly equal chips, wrong winner,
    # tiny confidence) -> the weakest-first rescue must recover it
    iq2 = iq.copy()
    flip_bit = 60
    a = pos + 16 + 2 * flip_bit
    if msg[flip_bit]:                     # true 1 (c0>c1) -> decode as 0
        iq2[a], iq2[a + 1] = 0.24 + 0j, 0.26 + 0j
    else:                                 # true 0 -> decode as 1
        iq2[a], iq2[a + 1] = 0.26 + 0j, 0.24 + 0j
    res2 = analyze(iq2, do_rescue=True)
    print(f"    1-bit corrupted: crc_ok={res2['crc_ok']} rescued={res2['rescued']}"
          f"  {'OK' if res2['rescued'] == 1 else 'FAIL'}")
    ok &= res2["rescued"] == 1
    print("=" * 62)
    print("SELFTEST", "PASS" if ok else "FAIL")
    print("=" * 62)
    return 0 if ok else 1


def cmd_capture(args):
    print(f"[capture] {args.secs:.0f}s live @ 1090 MHz on {args.antenna} ...")
    t0 = time.time()
    iq = capture_iq(args.secs, args.antenna, args.gain)
    if getattr(args, "save_iq", False):
        import json as _json
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = LAB / f"adsb_{stamp}.cs16"
        (np.round(np.column_stack([iq.real, iq.imag]).ravel() * 32767)
         .astype(np.int16)).tofile(out)
        _json.dump({"freq_hz": FREQ, "fs_hz": FS, "format": "cs16",
                    "n_samples": len(iq)}, open(str(out) + ".json", "w"))
        print(f"[corpus] saved {out.name} ({len(iq)*4/1e6:.0f} MB) - H1/H2 replay material")
    print(f"[capture] {len(iq)/FS:.1f}s captured, analyzing ...")
    res = analyze(iq)
    dt = time.time() - t0
    print(f"[result] candidates={res['candidates']}  CRC-valid={res['crc_ok']}"
          f"  rescued=+{res['rescued']}  in {dt:.0f}s wall")
    if res["planes"]:
        print(f"[planes] {len(res['planes'])} aircraft heard:")
        for icao, p in sorted(res["planes"].items(),
                              key=lambda kv: -kv[1]["msgs"])[:15]:
            cs = p.get("callsign", "-")
            alt = f"{p['alt_ft']} ft" if "alt_ft" in p else "-"
            spd = f"{p['speed_kt']} kt" if "speed_kt" in p else "-"
            print(f"    {icao}  msgs={p['msgs']:<4} callsign={cs:<9} "
                  f"alt={alt:<9} speed={spd}")
    else:
        print("[planes] none decoded - check antenna port / gain")
    return res


def cmd_shootout(args):
    print(f"[shootout] {args.secs:.0f}s per port - dial = CRC-valid messages")
    scores = {}
    for ant in ("Antenna A", "Antenna B", "Antenna C"):
        try:
            iq = capture_iq(args.secs, ant, args.gain)
        except Exception as e:
            print(f"  {ant}: capture failed ({e})")
            continue
        res = analyze(iq)
        scores[ant] = res
        print(f"  {ant}: crc_ok={res['crc_ok']} (+{res['rescued']} rescued) "
              f"planes={len(res['planes'])}")
        time.sleep(0.5)
    if scores:
        best = max(scores, key=lambda a: scores[a]["crc_ok"])
        print(f"[shootout] WINNER: {best} "
              f"({scores[best]['crc_ok']} valid msgs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    c = sub.add_parser("capture")
    c.add_argument("--secs", type=float, default=20)
    c.add_argument("--antenna", default="Antenna B")
    c.add_argument("--gain", type=float, default=45)
    c.add_argument("--save-iq", action="store_true",
                   help="archive the raw IQ for replay A/B (H1/H2 corpus)")
    s = sub.add_parser("shootout")
    s.add_argument("--secs", type=float, default=15)
    s.add_argument("--gain", type=float, default=45)
    args = ap.parse_args()
    if args.cmd == "selftest":
        sys.exit(cmd_selftest(args))
    elif args.cmd == "capture":
        cmd_capture(args)
    elif args.cmd == "shootout":
        cmd_shootout(args)


if __name__ == "__main__":
    main()
