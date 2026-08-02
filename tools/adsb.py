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

Fleet citizenship: capture paths acquire the one-radio radio_lock when the
fleet's lock module is present (fleet_lock()) - never a bare open.

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
                df = (bits[nmsg, 0] << 4) | (bits[nmsg, 1] << 3) \
                    | (bits[nmsg, 2] << 2) | (bits[nmsg, 3] << 1) \
                    | bits[nmsg, 4]
                nmsg += 1
                # short Mode S replies (DF 0/4/5/11) occupy 56 bits -
                # skip only their footprint so a reply right behind
                # them is not swallowed
                if df == 0 or df == 4 or df == 5 or df == 11:
                    i += 128
                else:
                    i += 240
                continue
        i += 1
    return starts[:nmsg], bits[:nmsg], conf[:nmsg]


if _HAVE_NUMBA:
    _scan = njit(cache=True)(_scan_impl)
else:
    _scan = _scan_impl


def demod_frames(iq, thresh_ratio=1.5):
    """IQ (complex64 @ 2 MS/s) -> list of dicts with bits/conf/CRC status.

    kinds: 'es'   DF17/18 extended squitter (112b, CRC-verifiable, rescuable)
           'df11' all-call reply (56b; remainder 0 = acquisition squitter,
                  ICAO carried in the clear)
           'ap'   DF4/5/20/21 interrogation replies: parity is XORed with
                  the ICAO address, so crc24() RECOVERS the address - it
                  can only be trusted against a whitelist of aircraft
                  already heard (the dump1090 approach). Never rescued."""
    mag = np.abs(iq).astype(np.float32)
    floor = 3.5 * float(np.median(mag))
    starts, bits, conf = _scan(mag, _PRE_HI, _PRE_LO, thresh_ratio, floor)
    out = []
    for k in range(len(starts)):
        b = bits[k]
        df = (int(b[0]) << 4) | (int(b[1]) << 3) | (int(b[2]) << 2) \
            | (int(b[3]) << 1) | int(b[4])
        if df in (17, 18):
            rem = crc24(b)
            out.append({"start": int(starts[k]), "df": df, "kind": "es",
                        "bits": b, "conf": conf[k],
                        "crc_ok": rem == 0, "rem": rem})
        elif df == 11:
            b56 = b[:56]
            rem = crc24(b56)
            out.append({"start": int(starts[k]), "df": 11, "kind": "df11",
                        "bits": b56, "conf": conf[k][:56],
                        "crc_ok": rem == 0, "rem": rem})
        elif df in (4, 5, 20, 21):
            n = 56 if df in (4, 5) else 112
            bn = b[:n]
            out.append({"start": int(starts[k]), "df": df, "kind": "ap",
                        "bits": bn, "conf": conf[k][:n],
                        "crc_ok": False, "addr": f"{crc24(bn):06X}"})
    return out


# ==========================================================================
# Mode S reply fields (DF4/5/20/21 + DF11)
# ==========================================================================
def ac13_alt(bits):
    """13-bit altitude code at message bits 19..31 (DF4/DF20).
    M(b7)=metric and Gillham (Q=0) forms return None (v1)."""
    if int(bits[25]):                       # M: metric altitude
        return None
    if not int(bits[27]):                   # Q=0: 100 ft Gillham code
        return None
    n = (_bf(bits, 19, 25) << 5) | (int(bits[26]) << 4) | _bf(bits, 28, 32)
    return n * 25 - 1000


def id13_squawk(bits):
    """13-bit identity code at message bits 19..31 (DF5/DF21) -> squawk."""
    c1, a1, c2, a2, c4, a4, _x, b1, d1, b2, d2, b4, d4 = (
        int(bits[19 + i]) for i in range(13))
    return (f"{a4 * 4 + a2 * 2 + a1}{b4 * 4 + b2 * 2 + b1}"
            f"{c4 * 4 + c2 * 2 + c1}{d4 * 4 + d2 * 2 + d1}")


def decode_modes_fields(bits, df):
    """Field decode for interrogation replies (whitelist-matched 'ap')."""
    info = {}
    if df in (4, 20):
        alt = ac13_alt(bits)
        if alt is not None and -1000 <= alt <= 60000:
            info["alt_ft"] = alt
    elif df in (5, 21):
        info["squawk"] = id13_squawk(bits)
    if df in (20, 21) and _bf(bits, 32, 40) == 0x20:   # Comm-B BDS 2,0
        cs = ""
        for k in range(8):
            cs += _CHARSET[_bf(bits, 40 + 6 * k, 46 + 6 * k)]
        cs = cs.replace("#", "").strip()
        if cs and all(c.isalnum() or c == " " for c in cs):
            info["callsign"] = cs
    return info


def rescue_blind(bits, max_flips=2):
    """The dump1090-style baseline: exhaustive single-bit scan, then
    adjacent double-bit scan - no confidence, pure CRC search. Returns
    (fixed_bits, n_flips, n_tries)."""
    tries = 0
    for i in range(112):
        b2 = bits.copy()
        b2[i] ^= 1
        tries += 1
        if crc24(b2) == 0:
            return b2, 1, tries
    if max_flips >= 2:
        for i in range(111):
            b2 = bits.copy()
            b2[i] ^= 1
            b2[i + 1] ^= 1
            tries += 1
            if crc24(b2) == 0:
                return b2, 2, tries
    return None, 0, tries


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
    elif 9 <= tc <= 18 or 20 <= tc <= 22:   # airborne position
        if 9 <= tc <= 18:              # barometric altitude (Q-bit form)
            q = int(bits[47])
            if q:
                n = (_bf(bits, 40, 47) << 4) | _bf(bits, 48, 52)
                info["alt_ft"] = n * 25 - 1000
        info["cpr_odd"] = int(bits[53])
        info["lat_cpr"] = _bf(bits, 54, 71) / 131072.0
        info["lon_cpr"] = _bf(bits, 71, 88) / 131072.0
    elif tc == 19:                     # velocity
        st = _bf(bits, 37, 40)
        if st in (1, 2):               # ground speed (2 = supersonic, x4)
            vew = _bf(bits, 46, 56)
            vns = _bf(bits, 57, 67)
            if vew and vns:
                mul = 4 if st == 2 else 1
                vx = (vew - 1) * mul * (-1 if bits[45] else 1)   # east +
                vy = (vns - 1) * mul * (-1 if bits[56] else 1)   # north +
                info["speed_kt"] = int(round(math.hypot(vx, vy)))
                info["track_deg"] = (math.degrees(math.atan2(vx, vy))
                                     + 360.0) % 360.0
        elif st in (3, 4):             # airspeed + magnetic heading
            if int(bits[45]):
                info["track_deg"] = _bf(bits, 46, 56) * 360.0 / 1024.0
            aspd = _bf(bits, 57, 67)
            if aspd:
                info["speed_kt"] = (aspd - 1) * (4 if st == 4 else 1)
        vr = _bf(bits, 69, 78)
        if vr:
            info["vr_fpm"] = (vr - 1) * 64 * (-1 if bits[68] else 1)
    return info


# ==========================================================================
# CPR position decode (the v2 lever: lat/lon for the ATC scope)
# ==========================================================================
_NZ = 15


def cpr_nl(lat):
    """Number of longitude zones at a latitude (the standard NL function)."""
    a = abs(lat)
    if a >= 87.0:
        return 1 if a > 87.0 else 2
    x = 1.0 - math.cos(math.pi / (2.0 * _NZ))
    c = math.cos(math.radians(lat)) ** 2
    return int(math.floor(2.0 * math.pi / math.acos(1.0 - x / c)))


def cpr_global(lat_e, lon_e, lat_o, lon_o, newest_odd):
    """Unambiguous airborne position from an even/odd CPR pair (fields
    already scaled to [0,1)). Returns (lat, lon) or None when the pair
    straddles a latitude-zone boundary (caller waits for the next pair)."""
    dlat_e, dlat_o = 360.0 / 60.0, 360.0 / 59.0
    j = int(math.floor(59.0 * lat_e - 60.0 * lat_o + 0.5))
    lat_ev = dlat_e * ((j % 60) + lat_e)
    lat_od = dlat_o * ((j % 59) + lat_o)
    if lat_ev >= 270.0:
        lat_ev -= 360.0
    if lat_od >= 270.0:
        lat_od -= 360.0
    if cpr_nl(lat_ev) != cpr_nl(lat_od):
        return None
    nl = cpr_nl(lat_od if newest_odd else lat_ev)
    m = int(math.floor(lon_e * (nl - 1) - lon_o * nl + 0.5))
    if newest_odd:
        lat = lat_od
        ni = max(nl - 1, 1)
        lon = (360.0 / ni) * ((m % ni) + lon_o)
    else:
        lat = lat_ev
        ni = max(nl, 1)
        lon = (360.0 / ni) * ((m % ni) + lon_e)
    if lon >= 180.0:
        lon -= 360.0
    if not -90.0 <= lat <= 90.0:
        return None
    return lat, lon


def cpr_local(lat_cpr, lon_cpr, odd, ref_lat, ref_lon):
    """Single-message decode near a known reference (< ~180 nm): the
    aircraft's own last fix, once it has one."""
    dlat = 360.0 / 59.0 if odd else 360.0 / 60.0
    j = math.floor(ref_lat / dlat) + math.floor(
        0.5 + (ref_lat % dlat) / dlat - lat_cpr)
    lat = dlat * (j + lat_cpr)
    nl = max(cpr_nl(lat) - (1 if odd else 0), 1)
    dlon = 360.0 / nl
    m = math.floor(ref_lon / dlon) + math.floor(
        0.5 + (ref_lon % dlon) / dlon - lon_cpr)
    lon = dlon * (m + lon_cpr)
    return lat, lon


# ==========================================================================
# SDR capture
# ==========================================================================
def fleet_lock():
    """Optional citizenship in the one-radio fleet lock: cooperate when the
    lock module exists (a fresh clone has no fleet to collide with)."""
    cands = [os.environ.get("RADIO_LOCK_PY", ""),
             r"Z:\src\gr-radiotuna\tools\radio_lock.py"]
    for c in cands:
        if c and Path(c).is_file():
            sys.path.insert(0, str(Path(c).parent))
            try:
                import radio_lock
                return radio_lock
            except Exception:
                pass
    return None


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
    """One-shot capture. NEVER a bare open when a fleet lock exists:
    priority 60 = user-driven run; abort loudly if outranked."""
    rl = fleet_lock()
    if rl:
        if not rl.acquire("adsb_capture", "one-shot 1090 capture", 60,
                          wait_s=15.0):
            busy = rl.status() or {}
            raise RuntimeError(
                f"radio held by {busy.get('owner', '?')} "
                f"(p{busy.get('priority', '?')}) - not opening over it")
    try:
        return _capture_iq_locked(secs, antenna, gain_db)
    finally:
        if rl:
            rl.release("adsb_capture")


def _capture_iq_locked(secs, antenna, gain_db=45):
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
    es = [f for f in frames if f["kind"] == "es"]
    good = [f for f in es if f["crc_ok"]]
    rescued = []
    if do_rescue:
        for f in es:
            if not f["crc_ok"]:
                b2, nf = rescue(f["bits"], f["conf"])
                if b2 is not None:
                    rescued.append({**f, "bits": b2, "flips": nf})
    planes = {}
    for f in sorted(good + rescued, key=lambda f: f["start"]):
        d = decode_fields(f["bits"])
        p = planes.setdefault(d["icao"], {"msgs": 0, "_cpr": {}})
        p["msgs"] += 1
        for key in ("callsign", "alt_ft", "speed_kt", "track_deg", "vr_fpm"):
            if key in d:
                p[key] = d[key]
        if "lat_cpr" in d:                      # pair even/odd within capture
            t = f["start"] / FS
            odd = d["cpr_odd"]
            p["_cpr"][odd] = (d["lat_cpr"], d["lon_cpr"], t)
            if 0 in p["_cpr"] and 1 in p["_cpr"]:
                e, o = p["_cpr"][0], p["_cpr"][1]
                if abs(e[2] - o[2]) <= 10.0:
                    g = cpr_global(e[0], e[1], o[0], o[1],
                                   newest_odd=o[2] > e[2])
                    if g:
                        p["lat"], p["lon"] = round(g[0], 5), round(g[1], 5)
    # DF11 all-calls: aircraft announcing themselves outside ADS-B
    n11 = 0
    for f in (f for f in frames if f["kind"] == "df11" and f["crc_ok"]):
        icao = f"{_bf(f['bits'], 8, 32):06X}"
        planes.setdefault(icao, {"msgs": 0, "_cpr": {}})["msgs"] += 1
        n11 += 1
    # DF4/5/20/21: address recovered from parity, trusted only against
    # aircraft already heard (whitelist) - never against thin air
    n_ap = 0
    for f in (f for f in frames if f["kind"] == "ap"):
        p = planes.get(f["addr"])
        if p is None:
            continue
        p["msgs"] += 1
        n_ap += 1
        for k, v in decode_modes_fields(f["bits"], f["df"]).items():
            p.setdefault(k, v)
    for p in planes.values():
        p.pop("_cpr", None)
    return {"candidates": len(frames), "crc_ok": len(good),
            "rescued": len(rescued), "df11": n11, "modes_ap": n_ap,
            "planes": planes}


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
    # 2b. CPR position: the mode-s.org even/odd pair with published truth
    print("[2b] CPR global + local decode")
    de = decode_fields(hex_to_bits("8D40621D58C382D690C8AC2863A7"))  # even
    do = decode_fields(hex_to_bits("8D40621D58C386435CC412692AD6"))  # odd
    g = cpr_global(de["lat_cpr"], de["lon_cpr"], do["lat_cpr"],
                   do["lon_cpr"], newest_odd=False)
    want = (52.2572021484375, 3.919372558593750)
    hit = g is not None and abs(g[0] - want[0]) < 1e-6 \
        and abs(g[1] - want[1]) < 1e-6
    print(f"    global -> {g}  {'OK' if hit else 'FAIL'}")
    ok &= hit
    loc = cpr_local(de["lat_cpr"], de["lon_cpr"], de["cpr_odd"], 52.26, 3.92)
    hit = abs(loc[0] - want[0]) < 1e-6 and abs(loc[1] - want[1]) < 1e-6
    print(f"    local  -> ({loc[0]:.7f}, {loc[1]:.7f})  {'OK' if hit else 'FAIL'}")
    ok &= hit
    # 2c. velocity vector: the mode-s.org groundspeed vector (159 kt, 182.9 deg)
    dv = decode_fields(hex_to_bits("8D485020994409940838175B284F"))
    hit = dv.get("speed_kt") == 159 and abs(dv.get("track_deg", 0) - 182.88) < 0.1 \
        and dv.get("vr_fpm") == -832
    print(f"    velocity vector -> {dv.get('speed_kt')} kt / "
          f"{dv.get('track_deg', 0):.1f} deg / {dv.get('vr_fpm')} fpm"
          f"  {'OK' if hit else 'FAIL'}")
    ok &= hit
    # 2d. Mode S replies: encoder/decoder roundtrips (machinery proof;
    # live consistency vs ADS-B altitudes is the field check)
    print("[2d] Mode S replies: DF11 / DF4 altitude / DF5 squawk")
    icao_t = 0xA1B2C3

    def _put(bb, a, b, val):
        for i in range(b - a):
            bb[a + i] = (val >> (b - a - 1 - i)) & 1
    b11 = np.zeros(56, np.uint8)
    _put(b11, 0, 5, 11)
    _put(b11, 8, 32, icao_t)
    _put(b11, 32, 56, crc24(b11))
    hit = crc24(b11) == 0 and _bf(b11, 8, 32) == icao_t
    print(f"    DF11 all-call: rem={crc24(b11)} icao={_bf(b11, 8, 32):06X}"
          f"  {'OK' if hit else 'FAIL'}")
    ok &= hit
    b4 = np.zeros(56, np.uint8)
    _put(b4, 0, 5, 4)
    _put(b4, 19, 25, 48)                  # AC13 for 38000 ft (N=1560, Q=1)
    b4[26] = 1
    b4[27] = 1
    _put(b4, 28, 32, 8)
    _put(b4, 32, 56, crc24(b4) ^ icao_t)  # address-parity
    hit = crc24(b4) == icao_t and ac13_alt(b4) == 38000
    print(f"    DF4 reply: addr={crc24(b4):06X} alt={ac13_alt(b4)} ft"
          f"  {'OK' if hit else 'FAIL'}")
    ok &= hit
    b5 = np.zeros(56, np.uint8)
    _put(b5, 0, 5, 5)
    for i, v in enumerate([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]):
        b5[19 + i] = v                    # squawk 7700
    _put(b5, 32, 56, crc24(b5) ^ icao_t)
    hit = crc24(b5) == icao_t and id13_squawk(b5) == "7700"
    print(f"    DF5 reply: addr={crc24(b5):06X} squawk={id13_squawk(b5)}"
          f"  {'OK' if hit else 'FAIL'}")
    ok &= hit
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
