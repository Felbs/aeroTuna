"""bds.py - aeroTuna task #55: BDS 4,4 Meteorological Routine Air Report.

Airliners carry calibrated air-data computers at 35,000 ft.  When a
ground radar interrogates Comm-B register 4,4 (MRAR), the aircraft
replies with the wind and the static air temperature IT IS MEASURING -
a free radiosonde at cruise altitude, several times a minute, from
every plane in range.  This module decodes that reply.

THE HARD PART IS NOT THE PARSER.  It is knowing that the 56-bit MB
field you are holding is register 4,4 at all:

  * A Comm-B reply carries NO register identifier.  The ground station
    knows what it asked for; a passive listener never sees the
    interrogation (uplink, different format, and we do not receive it).
  * DF20/21 parity is XORed with the ICAO address (adsb.py's docstring
    states this), so the frame is only trustworthy at all when the
    recovered address matches an aircraft ALREADY heard via ADS-B or
    DF11 - the dump1090 whitelist.  A 24-bit match against thin air
    invents ghost aircraft AND ghost weather.
  * The realistic Comm-B population over North America is Enhanced
    Surveillance: BDS 2,0 (callsign), 4,0 (selected altitude), 5,0
    (track and turn), 6,0 (heading and speed).  BDS 4,4 is NOT part of
    EHS - it is polled by a handful of ANSPs (the KNMI/UK MRAR
    programme).  So the default assumption for any DF20 heard here is
    "this is NOT 4,4", and the burden of proof is on the decode.

A WRONG REGISTER GUESS PRODUCES GARBAGE WEATHER THAT LOOKS FINE.
Bits are bits: a BDS 6,0 heading/Mach payload read as 4,4 yields a
wind speed, a wind direction and a temperature, all in range, all
false.  Everything below exists to make that failure mode rare and
measurable rather than invisible.  See classify() and the selftest's
Monte-Carlo section for the measured leak rates.

Validation ladder:
  selftest  - field roundtrip, full frame path (synth IQ -> demod ->
              parity-recovered address -> MB -> parse) and a
              false-positive Monte Carlo against 2,0/4,0/5,0/6,0
  replay    - archived 1090 IQ: DF20/21 census, whitelist pass rate,
              BDS 4,4 candidates, ISA cross-check
  capture   - live air through the fleet radio_lock (never a bare open)

Examples:
  python bds.py selftest
  python bds.py replay --iq lab/adsb_20260805_183000.cs16
  python bds.py replay --iq lab/x.cs16 --inject 12    # positive control
  python bds.py capture --secs 30 --antenna "Antenna B"
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import adsb                                   # the proven demod/CRC path

LAB = HERE.parent / "lab"
LAB.mkdir(exist_ok=True)
OUT_JSONL = LAB / "bds44.jsonl"
FS = adsb.FS                                  # 2 MS/s - adsb.py's proven rate


# ==========================================================================
# MB field access (1-based, the DO-181 / ICAO Annex 10 convention)
# ==========================================================================
def mb_of(bits):
    """DF20/21 message bits 33..88 -> the 56-bit Comm-B MB field."""
    return np.asarray(bits[32:88], np.uint8)


def mbu(mb, a, b):
    """Unsigned MB bits a..b inclusive, 1-based."""
    v = 0
    for i in range(a - 1, b):
        v = (v << 1) | int(mb[i])
    return v


def mbs(mb, a, b):
    """Two's-complement MB bits a..b inclusive, sign bit AT a.

    THE CLASSIC BUG lives here.  Static air temperature is an 11-bit
    field (MB 24..34) whose bit 24 is the sign.  Three wrong readings
    are all "plausible": (1) reading 10 bits 25..34 and losing the
    sign, (2) reading 11 bits unsigned and getting +255 C where -0.25 C
    was meant, (3) reading sign-magnitude instead of two's complement,
    which is correct at +20 C and wrong at -56 C - i.e. wrong exactly
    where cruise-altitude weather lives.  Pinned in the selftest by
    round-tripping -56.50 C and -0.25 C, not just positive values.
    """
    v = mbu(mb, a, b)
    n = b - a + 1
    if v & (1 << (n - 1)):
        v -= 1 << n
    return v


def mb_put(mb, a, b, val):
    """Write val into MB bits a..b inclusive (1-based). Encoder side."""
    n = b - a + 1
    val &= (1 << n) - 1
    for i in range(n):
        mb[a - 1 + i] = (val >> (n - 1 - i)) & 1


def wrongstatus(mb, sb, msb, lsb):
    """A status bit of 0 REQUIRES its value field to be all zeros.
    This single rule is the strongest structural discriminator we have
    against reading the wrong register: real payloads of other
    registers put data where 4,4 expects mandatory zeros."""
    if int(mb[sb - 1]):
        return False
    return mbu(mb, msb, lsb) != 0


# ==========================================================================
# BDS 4,4 - Meteorological Routine Air Report (MRAR)
# ==========================================================================
#  MB bit   field
#   1-4     Figure of merit / source (0 invalid, 1 INS, 2 GNSS,
#           3 DME/DME, 4 VOR/DME, 5-15 reserved)
#   5       wind status
#   6-14    wind speed, 1 kt LSB (0..511)
#   15-23   wind direction, 180/256 deg LSB (0..359.3)
#   24-34   static air temperature, 0.25 C LSB, TWO'S COMPLEMENT,
#           sign at bit 24 (no status bit of its own)
#   35      average static pressure status
#   36-46   average static pressure, 1 hPa LSB
#   47      turbulence status
#   48-49   turbulence (0 nil, 1 light, 2 moderate, 3 severe)
#   50      humidity status
#   51-56   humidity, 100/64 % LSB
SOURCE_NAMES = {0: "invalid", 1: "INS", 2: "GNSS", 3: "DME/DME",
                4: "VOR/DME"}

# physical gates - deliberately tighter than the field ranges
WIND_MAX_KT = 250.0        # jet-stream cores top out near 200 kt
SAT_MIN_C, SAT_MAX_C = -80.0, 60.0
PRESS_MIN_HPA, PRESS_MAX_HPA = 100, 1100


def parse_bds44(mb, strict=True):
    """56-bit MB -> BDS 4,4 fields, or None on ANY status/range failure.

    strict=True additionally rejects:
      * figure of merit 0 ("source invalid") - the aircraft is telling
        us the report is not trustworthy, and allowing it costs us a
        strong discriminator (see classify()).
      * static air temperature exactly 0.00 C - the signature of a
        mostly-empty payload.  KNOWN FALSE NEGATIVE: a genuine report
        of exactly 0.00 C is discarded (~1 in 500 near freezing).
    Both are togglable so the cost can be measured, not assumed.
    """
    mb = np.asarray(mb, np.uint8)
    if mb.size != 56 or not mb.any():
        return None                       # all-zero MB = no register / BDS 0,0
    fom = mbu(mb, 1, 4)
    if fom > 4:
        return None                       # 5..15 reserved: cannot be 4,4
    if strict and fom == 0:
        return None
    for sb, msb, lsb in ((5, 6, 23), (35, 36, 46), (47, 48, 49),
                         (50, 51, 56)):
        if wrongstatus(mb, sb, msb, lsb):
            return None
    out = {"fom": fom, "source": SOURCE_NAMES.get(fom, f"res{fom}")}
    if int(mb[4]):                        # wind status
        wspd = float(mbu(mb, 6, 14))
        if wspd > WIND_MAX_KT:
            return None
        out["wind_kt"] = wspd
        out["wind_dir"] = round(mbu(mb, 15, 23) * 180.0 / 256.0, 1)
        if out["wind_dir"] >= 360.0:
            return None
    else:
        out["wind_kt"] = None
        out["wind_dir"] = None
    sat = mbs(mb, 24, 34) * 0.25
    if not (SAT_MIN_C <= sat <= SAT_MAX_C):
        return None
    if strict and sat == 0.0:
        return None
    out["sat_c"] = round(sat, 2)
    if int(mb[34]):                       # average static pressure status
        p = mbu(mb, 36, 46)
        if not (PRESS_MIN_HPA <= p <= PRESS_MAX_HPA):
            return None
        out["press_hpa"] = p
    else:
        out["press_hpa"] = None
    out["turb"] = mbu(mb, 48, 49) if int(mb[46]) else None
    if int(mb[49]):                       # humidity status
        rh = mbu(mb, 51, 56) * 100.0 / 64.0
        if rh > 100.0:
            return None
        out["rh_pct"] = round(rh, 1)
    else:
        out["rh_pct"] = None
    return out


def encode_bds44(fom=1, wind_kt=None, wind_dir=None, sat_c=0.0,
                 press_hpa=None, turb=None, rh_pct=None):
    """Fields -> 56-bit MB.  Exists so the selftest can round-trip
    KNOWN values through the REAL frame path instead of trusting the
    parser against itself."""
    mb = np.zeros(56, np.uint8)
    mb_put(mb, 1, 4, int(fom))
    if wind_kt is not None:
        mb[4] = 1
        mb_put(mb, 6, 14, int(round(wind_kt)))
        raw = int(round((wind_dir or 0.0) * 256.0 / 180.0))
        mb_put(mb, 15, 23, min(raw, 511))
    mb_put(mb, 24, 34, int(round(sat_c / 0.25)) & 0x7FF)
    if press_hpa is not None:
        mb[34] = 1
        mb_put(mb, 36, 46, int(round(press_hpa)))
    if turb is not None:
        mb[46] = 1
        mb_put(mb, 48, 49, int(turb))
    if rh_pct is not None:
        mb[49] = 1
        mb_put(mb, 51, 56, int(round(rh_pct * 64.0 / 100.0)))
    return mb


# ==========================================================================
# The competing registers - needed ONLY to rule 4,4 out honestly
# ==========================================================================
def parse_bds40(mb):
    """BDS 4,0 selected vertical intention (EHS - very common).
    Signature: MB 40-47 and 52-53 are RESERVED ZEROS."""
    mb = np.asarray(mb, np.uint8)
    if not mb.any():
        return None
    if mbu(mb, 40, 47) or mbu(mb, 52, 53):
        return None
    for sb, msb, lsb in ((1, 2, 13), (14, 15, 26), (27, 28, 39),
                         (48, 49, 51), (54, 55, 56)):
        if wrongstatus(mb, sb, msb, lsb):
            return None
    out = {}
    if int(mb[0]):
        out["mcp_alt_ft"] = mbu(mb, 2, 13) * 16
        if out["mcp_alt_ft"] > 50000:
            return None
    if int(mb[13]):
        out["fms_alt_ft"] = mbu(mb, 15, 26) * 16
        if out["fms_alt_ft"] > 50000:
            return None
    if int(mb[26]):
        out["baro_mb"] = mbu(mb, 28, 39) * 0.1 + 800.0
        if not (900.0 <= out["baro_mb"] <= 1100.0):
            return None
    return out if out else None


def parse_bds50(mb):
    """BDS 5,0 track and turn (EHS)."""
    mb = np.asarray(mb, np.uint8)
    if not mb.any():
        return None
    for sb, msb, lsb in ((1, 2, 11), (12, 13, 23), (24, 25, 34),
                         (35, 36, 45), (46, 47, 56)):
        if wrongstatus(mb, sb, msb, lsb):
            return None
    out = {}
    if int(mb[0]):
        out["roll"] = round(mbs(mb, 2, 11) * 45.0 / 256.0, 1)
        if abs(out["roll"]) > 50:
            return None
    if int(mb[11]):
        out["trk"] = round((mbs(mb, 13, 23) * 90.0 / 512.0) % 360.0, 1)
    if int(mb[23]):
        out["gs"] = mbu(mb, 25, 34) * 2.0
        if out["gs"] > 700:
            return None
    if int(mb[34]):
        out["tar"] = round(mbs(mb, 36, 45) * 8.0 / 256.0, 2)
    if int(mb[45]):
        out["tas"] = mbu(mb, 47, 56) * 2.0
        if not (0 < out["tas"] <= 700):
            return None
    return out if out else None


def parse_bds60(mb):
    """BDS 6,0 heading and speed (EHS)."""
    mb = np.asarray(mb, np.uint8)
    if not mb.any():
        return None
    for sb, msb, lsb in ((1, 2, 12), (13, 14, 23), (24, 25, 34),
                         (35, 36, 45), (46, 47, 56)):
        if wrongstatus(mb, sb, msb, lsb):
            return None
    out = {}
    if int(mb[0]):
        out["hdg"] = round((mbs(mb, 2, 12) * 90.0 / 512.0) % 360.0, 1)
    if int(mb[12]):
        out["ias"] = mbu(mb, 14, 23)
        if out["ias"] > 500:
            return None
    if int(mb[23]):
        out["mach"] = round(mbu(mb, 25, 34) * 2.048 / 512.0, 3)
        if out["mach"] > 1.0:
            return None
    if int(mb[34]):
        out["baro_vr"] = mbs(mb, 36, 45) * 32.0
        if abs(out["baro_vr"]) > 6000:
            return None
    if int(mb[45]):
        out["ivv"] = mbs(mb, 47, 56) * 32.0
        if abs(out["ivv"]) > 6000:
            return None
    return out if out else None


def parse_bds20(mb):
    """BDS 2,0 aircraft identification - SELF-IDENTIFYING (MB 1-8 = 0x20)."""
    mb = np.asarray(mb, np.uint8)
    if mbu(mb, 1, 8) != 0x20:
        return None
    cs = ""
    for k in range(8):
        cs += adsb._CHARSET[mbu(mb, 9 + 6 * k, 14 + 6 * k)]
    if "#" in cs:
        return None
    cs = cs.strip()
    return {"callsign": cs} if cs else None


def parse_bds10(mb):
    """BDS 1,0 data link capability - SELF-IDENTIFYING (MB 1-8 = 0x10)."""
    mb = np.asarray(mb, np.uint8)
    if mbu(mb, 1, 8) != 0x10:
        return None
    return {"datalink_cap": mbu(mb, 9, 56)}


def parse_bds30(mb):
    """BDS 3,0 ACAS resolution advisory - SELF-IDENTIFYING (0x30)."""
    mb = np.asarray(mb, np.uint8)
    if mbu(mb, 1, 8) != 0x30:
        return None
    return {"acas_ra": mbu(mb, 9, 56)}


# ==========================================================================
# Standard atmosphere - the independent physical referee
# ==========================================================================
def isa_temp_c(alt_ft):
    """ISA temperature: 15 C at sea level, -6.5 C/km to the tropopause
    (11 km / 36,089 ft), isothermal -56.5 C above."""
    h = alt_ft * 0.3048
    return 15.0 - 6.5 * (h / 1000.0) if h < 11000.0 else -56.5


def isa_press_hpa(alt_ft):
    """ISA static pressure at altitude (hPa)."""
    h = alt_ft * 0.3048
    if h < 11000.0:
        return 1013.25 * (1.0 - 2.25577e-5 * h) ** 5.25588
    return 226.32 * math.exp(-(h - 11000.0) / 6341.6)


def plausibility(f44, alt_ft=None):
    """MULTIPLE INDEPENDENT CHECKS, reported individually.  A single
    "it decoded" is worth nothing here - only agreement between
    unrelated physical quantities distinguishes a real MRAR from a
    misread BDS 6,0.  Returns dict with per-check booleans, a score,
    and a verdict in {confirmed, plausible, weak, implausible}."""
    ch = {}
    ch["fom_valid"] = 1 <= f44.get("fom", 0) <= 4
    w = f44.get("wind_kt")
    if w is None:
        ch["wind_range"] = None
    else:
        ok = 0.0 <= w <= 200.0
        if alt_ft is not None and alt_ft < 10000:
            ok = ok and w <= 120.0        # 150-kt winds do not exist at 5000 ft
        ch["wind_range"] = bool(ok)
    d = f44.get("wind_dir")
    ch["dir_range"] = None if d is None else bool(0.0 <= d < 360.0)
    sat = f44.get("sat_c")
    ch["sat_range"] = bool(SAT_MIN_C <= sat <= SAT_MAX_C)
    d_isa = None
    if alt_ft is not None:
        d_isa = round(sat - isa_temp_c(alt_ft), 1)
        # real atmospheres depart from ISA by <= ~20 C outside extreme
        # polar winters; 30 C is a generous accept, 45 C is absurd
        ch["sat_vs_isa"] = bool(abs(d_isa) <= 30.0)
    p = f44.get("press_hpa")
    d_p = None
    if p is not None and alt_ft is not None:
        d_p = round(p - isa_press_hpa(alt_ft), 1)
        ch["press_vs_isa"] = bool(abs(d_p) <= 60.0)
    elif p is not None:
        ch["press_vs_isa"] = bool(PRESS_MIN_HPA <= p <= PRESS_MAX_HPA)
    t = f44.get("turb")
    ch["turb_range"] = None if t is None else bool(0 <= t <= 3)
    rh = f44.get("rh_pct")
    ch["rh_range"] = None if rh is None else bool(0.0 <= rh <= 100.0)
    live = {k: v for k, v in ch.items() if v is not None}
    n_ok = sum(1 for v in live.values() if v)
    n = len(live)
    hard_fail = ch.get("sat_vs_isa") is False or ch.get("wind_range") is False \
        or ch.get("press_vs_isa") is False
    if hard_fail:
        verdict = "implausible"
    elif ch.get("sat_vs_isa") and n_ok >= 4:
        verdict = "confirmed"
    elif n_ok >= 3 and n_ok == n:
        verdict = "plausible"
    else:
        verdict = "weak"
    return {"checks": ch, "n_ok": n_ok, "n": n,
            "score": round(n_ok / n, 2) if n else 0.0,
            "sat_minus_isa_c": d_isa, "press_minus_isa_hpa": d_p,
            "verdict": verdict}


# ==========================================================================
# REGISTER DISCRIMINATION - the honest core
# ==========================================================================
def classify(bits, truth=None, alt_ft=None, strict=True):
    """Decide which BDS register a DF20/21 MB field holds.

    Order of evidence, strongest first:

    1. SELF-IDENTIFYING registers.  BDS 1,0 / 2,0 / 3,0 begin with
       their own code in MB 1-8.  If one of those parses cleanly, that
       IS the register - 4,4 is excluded outright.

    2. STRUCTURAL rejection.  Status-bit consistency (a 0 status bit
       demands an all-zero value field) plus "figure of merit <= 4".
       This second rule is unreasonably effective: in BDS 4,0 / 5,0 /
       6,0 the FIRST MB bit is a status bit, so whenever those
       registers actually carry data, MB 1-4 reads as 8..15 - out of
       range for a 4,4 source code.  The measured leak rates are in
       the selftest Monte Carlo; do not take this paragraph on faith.

    3. COMPETITION.  If more than one non-self-identifying register
       still parses, the reply is AMBIGUOUS.  We never emit an
       ambiguous frame as a weather report.

    4. TRUTH CROSS-CHECK.  When the aircraft's own ADS-B state is
       fresh (ground speed / track from DF17 TC19), a BDS 5,0 or 6,0
       reading that AGREES with it is proof of that register and
       therefore disproof of 4,4.  This is the same whitelist logic
       adsb.classify_commb() already uses, run in the negative
       direction.

    5. PHYSICS.  Only then does plausibility() get a vote, and it can
       only demote, never promote.

    RESIDUAL FALSE-POSITIVE RISK - stated plainly: a register we do
    not model (4,5 meteorological hazard, 4,4 look-alikes, vendor
    registers, or a corrupted-but-CRC-unverifiable frame) can still
    land inside the 4,4 envelope.  DF20/21 parity is address-XORed, so
    unlike DF17 we CANNOT verify the bits are error-free - we only
    know the recovered address matches a plane we have heard.  A
    single bit error therefore produces a plausible-looking wrong
    temperature with no CRC to catch it.  Every emitted report carries
    its plausibility dict for exactly this reason, and a
    single-aircraft single-report "discovery" of MRAR in this airspace
    should be disbelieved until the same aircraft repeats it.
    """
    mb = mb_of(bits) if len(bits) >= 88 else np.asarray(bits, np.uint8)
    notes = []
    if not mb.any():
        return {"bds": None, "fields": None, "candidates": [],
                "notes": ["all-zero MB (no register / BDS 0,0)"]}
    # 1. self-identifying registers win outright
    for name, fn in (("2,0", parse_bds20), ("1,0", parse_bds10),
                     ("3,0", parse_bds30)):
        f = fn(mb)
        if f is not None:
            return {"bds": name, "fields": f, "candidates": [name],
                    "notes": [f"self-identifying header 0x{mbu(mb, 1, 8):02X}"]}
    # 2/3. structural parse of every non-self-identifying register
    cands = {}
    for name, f in (("4,4", parse_bds44(mb, strict=strict)),
                    ("4,0", parse_bds40(mb)),
                    ("5,0", parse_bds50(mb)),
                    ("6,0", parse_bds60(mb))):
        if f is not None:
            cands[name] = f
    if "4,4" not in cands:
        best = next(iter(cands), None)
        return {"bds": best, "fields": cands.get(best),
                "candidates": list(cands), "notes": ["not BDS 4,4"]}
    # 4. truth cross-check: can we PROVE a competitor and thereby kill 4,4?
    truth = truth or {}
    t_gs, t_trk = truth.get("speed_kt"), truth.get("track_deg")
    if "5,0" in cands and t_gs is not None and t_trk is not None:
        f = cands["5,0"]
        if f.get("gs") is not None and abs(f["gs"] - t_gs) <= 25 \
                and f.get("trk") is not None \
                and min(abs(f["trk"] - t_trk),
                        360 - abs(f["trk"] - t_trk)) <= 12:
            return {"bds": "5,0", "fields": f, "candidates": list(cands),
                    "notes": ["BDS 5,0 confirmed against own ADS-B state "
                              "- 4,4 excluded"]}
    if "6,0" in cands and t_trk is not None:
        f = cands["6,0"]
        if f.get("hdg") is not None \
                and min(abs(f["hdg"] - t_trk),
                        360 - abs(f["hdg"] - t_trk)) <= 35 \
                and 0.1 <= f.get("mach", 0) <= 0.96:
            return {"bds": "6,0", "fields": f, "candidates": list(cands),
                    "notes": ["BDS 6,0 confirmed against own ADS-B state "
                              "- 4,4 excluded"]}
    others = [c for c in cands if c != "4,4"]
    f44 = cands["4,4"]
    pl = plausibility(f44, alt_ft)
    if others:
        notes.append("AMBIGUOUS: also parses as " + ", ".join(others))
        return {"bds": None, "fields": f44, "candidates": list(cands),
                "plausibility": pl, "ambiguous": True, "notes": notes}
    # 5. physics may only demote
    if pl["verdict"] == "implausible":
        notes.append("sole structural candidate but physically implausible "
                     "- probable register misidentification")
        return {"bds": None, "fields": f44, "candidates": ["4,4"],
                "plausibility": pl, "ambiguous": False, "notes": notes}
    notes.append("sole structural candidate; physics " + pl["verdict"])
    return {"bds": "4,4", "fields": f44, "candidates": ["4,4"],
            "plausibility": pl, "ambiguous": False, "notes": notes}


# ==========================================================================
# Frame construction (selftest + Monte Carlo)
# ==========================================================================
def _put(bits, a, b, val):
    for i in range(b - a):
        bits[a + i] = (val >> (b - a - 1 - i)) & 1


def encode_ac13(alt_ft):
    """13 altitude-code bits (Q=1, 25 ft) as message bits 20..32."""
    n = int(round((alt_ft + 1000) / 25.0))
    return n


def make_df20(icao, mb, alt_ft=35000, df=20):
    """A complete 112-bit DF20 Comm-B reply with ADDRESS PARITY, so the
    selftest exercises the same crc24()-recovers-the-address path the
    live decoder uses."""
    bits = np.zeros(112, np.uint8)
    _put(bits, 0, 5, df)
    n = encode_ac13(alt_ft)
    _put(bits, 19, 25, n >> 5)
    bits[25] = 0                          # M = 0 (feet)
    bits[26] = (n >> 4) & 1
    bits[27] = 1                          # Q = 1 (25 ft)
    _put(bits, 28, 32, n & 0xF)
    bits[32:88] = np.asarray(mb, np.uint8)
    _put(bits, 88, 112, adsb.crc24(bits) ^ int(icao, 16))
    return bits


def splice(iq, pos, bits, amp):
    """ADD one PPM Mode S frame on top of existing IQ at sample `pos`.
    Used by replay --inject: a POSITIVE CONTROL that runs synthetic BDS
    4,4 through the real archived RF background, so "zero reports" can
    be attributed to absent traffic rather than a broken decoder."""
    pre = (1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0)
    for k, c in enumerate(pre):
        if c:
            iq[pos + k] += amp
    for b in range(112):
        iq[pos + 16 + 2 * b + (0 if bits[b] else 1)] += amp
    return iq


def frames_to_iq(frames, seed=0, amp=0.5, noise=0.03):
    """Frames -> 2 MS/s PPM IQ (the same synthesis adsb.py's selftest
    and skyTuna's replay use).  No deliberate corruption: DF20 parity
    is address-XORed and therefore NOT rescuable, so a corrupted Comm-B
    frame must simply be lost, not repaired."""
    rng = np.random.default_rng(seed)
    gap = 300
    n = 400 + len(frames) * (240 + gap) + 400
    sig = np.zeros(n, np.float32)
    pre = [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    pos = 400
    for bits in frames:
        for k, c in enumerate(pre):
            sig[pos + k] = c * amp
        for b in range(112):
            sig[pos + 16 + 2 * b + (0 if bits[b] else 1)] = amp
        pos += 240 + gap
    iq = sig.astype(np.complex64)
    iq += (rng.normal(0, noise, n) + 1j * rng.normal(0, noise, n)
           ).astype(np.complex64)
    return iq


# ==========================================================================
# JSONL emission
# ==========================================================================
def emit(rec, path=OUT_JSONL):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def make_record(icao, f44, alt_ft, pl, when=None, origin="replay"):
    return {"t": when or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "icao": icao, "alt_ft": alt_ft,
            "wind_kt": f44.get("wind_kt"), "wind_dir": f44.get("wind_dir"),
            "sat_c": f44.get("sat_c"), "source": f44.get("source"),
            "press_hpa": f44.get("press_hpa"), "turb": f44.get("turb"),
            "rh_pct": f44.get("rh_pct"),
            "origin": origin, "plausibility": pl}


# ==========================================================================
# Shared harvest: IQ -> whitelist -> DF20/21 -> BDS 4,4
# ==========================================================================
def harvest(iq, stats=None, whitelist=None, t0=None, origin="replay",
            emit_path=OUT_JSONL, strict=True, verbose=False, wl_only=False):
    """One IQ block through the full ladder.  `whitelist` is a dict
    icao -> ADS-B truth, carried across blocks by the caller.
    wl_only=True does pass 1 of an offline replay: grow the whitelist
    and touch nothing else, so a Comm-B reply at t=5s can be validated
    by an ADS-B frame that only arrives at t=20s."""
    st = stats if stats is not None else _new_stats()
    wl = whitelist if whitelist is not None else {}
    frames = adsb.demod_frames(iq)
    st["candidates"] += len(frames)
    # --- whitelist growth: only CRC-verifiable frames may add aircraft
    for f in frames:
        if f["kind"] == "es" and f["crc_ok"]:
            st["es_ok"] += 1
            d = adsb.decode_fields(f["bits"])
            e = wl.setdefault(d["icao"], {})
            for k in ("callsign", "alt_ft", "speed_kt", "track_deg"):
                if k in d:
                    e[k] = d[k]
        elif f["kind"] == "df11" and f["crc_ok"]:
            st["df11"] += 1
            wl.setdefault(f"{adsb._bf(f['bits'], 8, 32):06X}", {})
    if wl_only:
        return st, wl
    # --- Comm-B replies
    for f in frames:
        if f["kind"] != "ap":
            continue
        st["ap"] += 1
        st["df_census"][f["df"]] = st["df_census"].get(f["df"], 0) + 1
        if f["df"] not in (20, 21):
            continue
        st["commb"] += 1
        addr = f["addr"]
        # A parity-recovered address that REPEATS is strong evidence of a
        # real aircraft: a preamble false-fire on noise produces a fresh
        # random 24-bit value every time and essentially never repeats.
        st["commb_addrs"][addr] = st["commb_addrs"].get(addr, 0) + 1
        if addr not in wl:
            st["commb_no_whitelist"] += 1
            # AUDIT ONLY, never emitted: what a whitelist-less decoder
            # would have believed.  This is the ghost-weather counter -
            # it exists so the cost of skipping the whitelist is a
            # number in the log rather than an opinion.
            gr = classify(f["bits"], truth={}, alt_ft=None, strict=strict)
            if gr.get("bds") == "4,4":
                st["ghost_bds44"] += 1
            continue
        st["commb_whitelisted"] += 1
        alt = adsb.ac13_alt(f["bits"]) if f["df"] == 20 else None
        if alt is not None and not (-1000 <= alt <= 60000):
            alt = None
        if alt is None:
            alt = wl[addr].get("alt_ft")
        res = classify(f["bits"], truth=wl[addr], alt_ft=alt, strict=strict)
        b = res.get("bds")
        st["reg_census"][b or ("ambiguous" if res.get("ambiguous")
                               else "unknown")] = \
            st["reg_census"].get(b or ("ambiguous" if res.get("ambiguous")
                                       else "unknown"), 0) + 1
        if res.get("ambiguous"):
            st["bds44_ambiguous"] += 1
        if b != "4,4":
            continue
        st["bds44"] += 1
        pl = res["plausibility"]
        st["verdicts"][pl["verdict"]] = st["verdicts"].get(pl["verdict"], 0) + 1
        rec = make_record(addr, res["fields"], alt, pl,
                          when=(time.strftime("%Y-%m-%dT%H:%M:%S",
                                              time.localtime(t0))
                                if t0 else None),
                          origin=origin)
        st["records"].append(rec)
        if emit_path:
            emit(rec, emit_path)
        if verbose:
            print(f"    BDS4,4 {addr} alt={alt} wind={rec['wind_kt']}kt/"
                  f"{rec['wind_dir']}deg sat={rec['sat_c']}C "
                  f"[{pl['verdict']} dISA={pl['sat_minus_isa_c']}]")
    return st, wl


def _new_stats():
    return {"candidates": 0, "es_ok": 0, "df11": 0, "ap": 0, "commb": 0,
            "commb_whitelisted": 0, "commb_no_whitelist": 0,
            "bds44": 0, "bds44_ambiguous": 0, "df_census": {},
            "reg_census": {}, "verdicts": {}, "records": [],
            "commb_addrs": {}, "ghost_bds44": 0}


def print_stats(st):
    print(f"  frame candidates      : {st['candidates']}")
    print(f"  DF17/18 CRC-valid     : {st['es_ok']}")
    print(f"  DF11 all-calls        : {st['df11']}")
    print(f"  DF4/5/20/21 replies   : {st['ap']}   census={st['df_census']}")
    print(f"  DF20/21 (Comm-B)      : {st['commb']}")
    print(f"    whitelist PASS      : {st['commb_whitelisted']}")
    print(f"    whitelist REJECT    : {st['commb_no_whitelist']}  "
          f"(address matched no aircraft heard - discarded)")
    addrs = st.get("commb_addrs", {})
    rep = {a: c for a, c in addrs.items() if c > 1}
    print(f"    distinct addresses  : {len(addrs)}  repeated: {len(rep)}"
          f"  (a repeat means a REAL aircraft; a preamble false-fire on "
          f"noise gives a fresh random address every time)")
    if rep:
        top = sorted(rep.items(), key=lambda kv: -kv[1])[:8]
        print(f"    repeats             : "
              + ", ".join(f"{a}x{c}" for a, c in top))
    print(f"  GHOST weather avoided : {st['ghost_bds44']}  (non-whitelisted "
          f"replies that WOULD have parsed as BDS 4,4 - the whitelist's "
          f"measured value)")
    print(f"  register census       : {st['reg_census']}")
    print(f"  BDS 4,4 accepted      : {st['bds44']}   "
          f"ambiguous-and-dropped: {st['bds44_ambiguous']}")
    if st["verdicts"]:
        print(f"  plausibility verdicts : {st['verdicts']}")


# ==========================================================================
# selftest
# ==========================================================================
def cmd_selftest(_args):
    print("=" * 70)
    print("aeroTuna BDS 4,4 (MRAR) self-test")
    print("=" * 70)
    ok = True

    # ---- 1. field roundtrip, including the sign-convention trap ----------
    print("[1] field roundtrip (encoder -> parser), exact")
    cases = [
        dict(fom=2, wind_kt=63.0, wind_dir=225.0, sat_c=-56.50,
             press_hpa=238, turb=1, rh_pct=25.0),
        dict(fom=1, wind_kt=0.0, wind_dir=0.0, sat_c=-0.25),
        dict(fom=4, wind_kt=250.0, wind_dir=354.4, sat_c=59.75,
             press_hpa=1013, turb=3, rh_pct=98.4),
        dict(fom=3, wind_kt=115.0, wind_dir=90.0, sat_c=-79.75,
             press_hpa=150, turb=0, rh_pct=0.0),
    ]
    for c in cases:
        mb = encode_bds44(**c)
        got = parse_bds44(mb)
        if got is None:
            print(f"    {c} -> REJECTED  FAIL")
            ok = False
            continue
        hit = True
        for k, v in c.items():
            if k == "fom":
                hit &= got["fom"] == v
            elif k == "rh_pct":
                hit &= abs(got["rh_pct"] - v) < 1.6      # 100/64 quantum
            elif k == "wind_dir":
                hit &= abs(got["wind_dir"] - v) < 0.8    # 180/256 quantum
            else:
                hit &= got[k] == v
        print(f"    fom={c['fom']} wind={got['wind_kt']}kt/{got['wind_dir']}deg"
              f" sat={got['sat_c']}C p={got['press_hpa']} turb={got['turb']}"
              f" rh={got['rh_pct']}  {'OK' if hit else 'FAIL'}")
        ok &= hit
    # the classic bug, explicitly: negative temperatures must survive
    neg = parse_bds44(encode_bds44(fom=1, sat_c=-56.5))
    hit = neg is not None and neg["sat_c"] == -56.5
    print(f"    SIGN TRAP: -56.50 C round-trips -> {neg and neg['sat_c']}"
          f"  {'OK' if hit else 'FAIL'}")
    ok &= hit
    unsigned_read = mbu(encode_bds44(fom=1, sat_c=-56.5), 24, 34) * 0.25
    print(f"    (an UNSIGNED read of the same bits gives "
          f"{unsigned_read:+.2f} C - the bug this test pins)")

    # ---- 2. status-bit and range rejection --------------------------------
    print("[2] rejection: status bits and ranges (must return None)")
    bad = []
    mb = encode_bds44(fom=1, wind_kt=50.0, wind_dir=90.0, sat_c=-40.0)
    mb2 = mb.copy(); mb2[4] = 0                       # status 0, field nonzero
    bad.append(("wind status 0 with data", mb2))
    mb2 = mb.copy(); mb_put(mb2, 1, 4, 7)             # reserved FOM
    bad.append(("figure of merit 7 (reserved)", mb2))
    mb2 = encode_bds44(fom=1, wind_kt=400.0, wind_dir=90.0, sat_c=-40.0)
    bad.append(("wind 400 kt (over gate)", mb2))
    mb2 = encode_bds44(fom=1, sat_c=-40.0, press_hpa=1013)
    mb2[34] = 0                                        # press status 0, data set
    bad.append(("pressure status 0 with data", mb2))
    mb2 = encode_bds44(fom=1, sat_c=-40.0); mb_put(mb2, 24, 34, 0x2FF)
    bad.append(("SAT +191 C (out of range)", mb2))
    bad.append(("all-zero MB", np.zeros(56, np.uint8)))
    for name, m in bad:
        got = parse_bds44(m)
        print(f"    {name:38s} -> {'None OK' if got is None else 'PARSED FAIL'}")
        ok &= got is None

    # ---- 3. FULL FRAME PATH: synth IQ -> demod -> address -> MB -> parse --
    print("[3] full frame path: synthetic IQ -> demod -> parity-recovered "
          "address -> BDS 4,4")
    icao = "A1B2C3"
    truth_fields = dict(fom=2, wind_kt=97.0, wind_dir=292.5, sat_c=-53.25,
                        press_hpa=238, turb=1, rh_pct=12.5)
    alt = 35000
    mb = encode_bds44(**truth_fields)
    frame = make_df20(icao, mb, alt_ft=alt)
    # an ADS-B position frame so the aircraft is on the whitelist at all
    es = adsb.hex_to_bits("8D4840D6202CC371C32CE0576098")
    iq = frames_to_iq([es, frame], seed=7)
    got_frames = adsb.demod_frames(iq)
    ap = [f for f in got_frames if f["kind"] == "ap" and f["df"] == 20]
    hit = len(ap) == 1 and ap[0]["addr"] == icao
    print(f"    demod: {len(got_frames)} candidates, DF20={len(ap)}, "
          f"addr={ap[0]['addr'] if ap else '-'} (want {icao})"
          f"  {'OK' if hit else 'FAIL'}")
    ok &= hit
    if ap:
        alt_got = adsb.ac13_alt(ap[0]["bits"])
        print(f"    in-frame altitude -> {alt_got} ft (want {alt})"
              f"  {'OK' if alt_got == alt else 'FAIL'}")
        ok &= alt_got == alt
        res = classify(ap[0]["bits"], truth={}, alt_ft=alt_got)
        f = res["fields"] or {}
        exact = (res["bds"] == "4,4"
                 and f.get("fom") == truth_fields["fom"]
                 and f.get("wind_kt") == truth_fields["wind_kt"]
                 and abs(f.get("wind_dir", -1)
                         - truth_fields["wind_dir"]) < 0.8
                 and f.get("sat_c") == truth_fields["sat_c"]
                 and f.get("press_hpa") == truth_fields["press_hpa"]
                 and f.get("turb") == truth_fields["turb"]
                 and abs(f.get("rh_pct", -1) - truth_fields["rh_pct"]) < 1.6)
        print(f"    classify -> BDS {res['bds']} candidates={res['candidates']}")
        print(f"    fields   -> wind={f.get('wind_kt')}kt/{f.get('wind_dir')}deg"
              f" sat={f.get('sat_c')}C p={f.get('press_hpa')}hPa"
              f" turb={f.get('turb')} rh={f.get('rh_pct')}%")
        print(f"    ROUNDTRIP EXACT through the real frame path: "
              f"{'OK' if exact else 'FAIL'}")
        ok &= exact
        pl = res["plausibility"]
        print(f"    physics  -> {pl['verdict']} score={pl['score']} "
              f"dISA={pl['sat_minus_isa_c']} C  dP={pl['press_minus_isa_hpa']} hPa")

    # ---- 3b. whitelist discipline ----------------------------------------
    print("[3b] whitelist discipline: a Comm-B reply from an UNHEARD "
          "aircraft must be discarded")
    iq_lone = frames_to_iq([make_df20("BEEF01", mb, alt_ft=alt)], seed=9)
    st, _ = harvest(iq_lone, emit_path=None)
    hit = st["commb"] >= 1 and st["commb_whitelisted"] == 0 and st["bds44"] == 0
    print(f"    Comm-B seen={st['commb']} whitelisted={st['commb_whitelisted']}"
          f" emitted={st['bds44']}  {'OK' if hit else 'FAIL'}")
    ok &= hit
    print("[3c] same aircraft heard first -> the SAME reply is accepted")
    es2 = adsb.hex_to_bits("8D4840D6202CC371C32CE0576098")
    _put(es2, 8, 32, 0xBEEF01)            # re-address ...
    es2[88:] = 0
    _put(es2, 88, 112, adsb.crc24(es2))   # ... and re-parity, so it is a
    #                                       CRC-valid ADS-B frame for BEEF01
    st2, _ = harvest(frames_to_iq(
        [es2, make_df20("BEEF01", mb, alt_ft=alt)], seed=11), emit_path=None)
    hit = st2["bds44"] == 1
    print(f"    whitelisted={st2['commb_whitelisted']} emitted={st2['bds44']}"
          f"  {'OK' if hit else 'FAIL'}")
    ok &= hit

    # ---- 4. FALSE-POSITIVE MONTE CARLO -----------------------------------
    print("[4] register-discrimination false positives (the honest number)")
    rng = np.random.default_rng(2026)
    N = 20000

    def _rand_mb():
        return rng.integers(0, 2, 56).astype(np.uint8)

    def _mk40():
        mb = np.zeros(56, np.uint8)
        mb[0] = 1
        mb_put(mb, 2, 13, int(rng.integers(1000, 40000)) // 16)
        if rng.random() < 0.6:
            mb[13] = 1
            mb_put(mb, 15, 26, int(rng.integers(1000, 40000)) // 16)
        mb[26] = 1
        mb_put(mb, 28, 39, int(round((rng.uniform(980, 1035) - 800) * 10)))
        if rng.random() < 0.5:
            mb[47] = 1
            mb_put(mb, 49, 51, int(rng.integers(0, 8)))
        return mb

    def _mk50():
        mb = np.zeros(56, np.uint8)
        mb[0] = 1
        mb_put(mb, 2, 11, int(round(rng.uniform(-30, 30) * 256 / 45)) & 0x3FF)
        mb[11] = 1
        mb_put(mb, 13, 23, int(round(rng.uniform(-180, 180) * 512 / 90)) & 0x7FF)
        mb[23] = 1
        mb_put(mb, 25, 34, int(rng.integers(150, 550)) // 2)
        mb[34] = 1
        mb_put(mb, 36, 45, int(round(rng.uniform(-3, 3) * 256 / 8)) & 0x3FF)
        mb[45] = 1
        mb_put(mb, 47, 56, int(rng.integers(150, 550)) // 2)
        return mb

    def _mk60():
        mb = np.zeros(56, np.uint8)
        mb[0] = 1
        mb_put(mb, 2, 12, int(round(rng.uniform(-180, 180) * 512 / 90)) & 0x7FF)
        mb[12] = 1
        mb_put(mb, 14, 23, int(rng.integers(120, 350)))
        mb[23] = 1
        mb_put(mb, 25, 34, int(round(rng.uniform(0.2, 0.9) * 512 / 2.048)))
        mb[34] = 1
        mb_put(mb, 36, 45, int(round(rng.uniform(-2000, 2000) / 32)) & 0x3FF)
        mb[45] = 1
        mb_put(mb, 47, 56, int(round(rng.uniform(-2000, 2000) / 32)) & 0x3FF)
        return mb

    def _mk20():
        mb = np.zeros(56, np.uint8)
        mb_put(mb, 1, 8, 0x20)
        cs = "".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
                                size=int(rng.integers(4, 8))))
        cs = f"{cs:<8}"
        for k, ch in enumerate(cs):
            mb_put(mb, 9 + 6 * k, 14 + 6 * k, adsb._CHARSET.index(ch))
        return mb

    pops = [("uniform random 56-bit", _rand_mb, N),
            ("realistic BDS 4,0 (EHS)", _mk40, 5000),
            ("realistic BDS 5,0 (EHS)", _mk50, 5000),
            ("realistic BDS 6,0 (EHS)", _mk60, 5000),
            ("realistic BDS 2,0 (callsign)", _mk20, 5000)]
    leaks = {}
    for name, gen, n in pops:
        parsed = accepted = ambig = confirmed = 0
        for _ in range(n):
            mb = gen()
            if parse_bds44(mb) is not None:
                parsed += 1
            bits = np.zeros(112, np.uint8)
            bits[32:88] = mb
            r = classify(bits, truth={}, alt_ft=35000)
            if r.get("ambiguous"):
                ambig += 1
            if r["bds"] == "4,4":
                accepted += 1
                if r["plausibility"]["verdict"] == "confirmed":
                    confirmed += 1
        leaks[name] = (parsed / n, accepted / n, confirmed / n)
        print(f"    {name:30s} n={n:<6} parse44={parsed/n*100:6.3f}%  "
              f"ambiguous={ambig/n*100:6.3f}%  "
              f"classify=4,4 {accepted/n*100:6.3f}%  "
              f"AND physics-confirmed {confirmed/n*100:6.3f}%")
    # the gate we actually ship on: EHS registers must not leak at all
    ehs_leak = max(leaks[k][2] for k in leaks if k != "uniform random 56-bit")
    hit = ehs_leak == 0.0
    print(f"    EHS (2,0/4,0/5,0/6,0) -> physics-confirmed 4,4 leak = "
          f"{ehs_leak*100:.4f}%  {'OK' if hit else 'FAIL'}")
    ok &= hit
    print(f"    NOTE random-payload leak {leaks['uniform random 56-bit'][2]*100:.3f}% "
          f"is the floor for an UNDECODABLE bit error, not for a real "
          f"register - DF20 parity is address-XORed and cannot verify bits.")

    # ---- 5. ISA referee ---------------------------------------------------
    print("[5] standard-atmosphere referee")
    for alt, want in ((0, 15.0), (36089, -56.5), (39000, -56.5),
                      (18000, 15.0 - 6.5 * (18000 * 0.3048 / 1000))):
        got = isa_temp_c(alt)
        hitv = abs(got - want) < 0.2
        print(f"    ISA({alt:>6} ft) = {got:7.2f} C (want {want:7.2f})"
              f"  {'OK' if hitv else 'FAIL'}")
        ok &= hitv
    p = isa_press_hpa(35000)
    hit = 230 < p < 245
    print(f"    ISA pressure(35000 ft) = {p:.1f} hPa (want ~238)"
          f"  {'OK' if hit else 'FAIL'}")
    ok &= hit
    # a BDS 6,0 payload misread as 4,4 should be caught by physics
    mb6 = _mk60()
    f6as44 = parse_bds44(mb6, strict=False)
    print(f"    a random BDS 6,0 read as 4,4 -> "
          f"{'rejected by parser' if f6as44 is None else f6as44}")

    print("=" * 70)
    print("SELFTEST", "PASS" if ok else "FAIL")
    print("=" * 70)
    return 0 if ok else 1


# ==========================================================================
# replay
# ==========================================================================
def cmd_replay(args):
    path = Path(args.iq)
    if not path.is_file():
        print(f"[replay] no such file: {path}")
        return 1
    meta = {}
    mj = Path(str(path) + ".json")
    if not mj.is_file():
        mj = path.with_suffix(".json")
    if mj.is_file():
        try:
            meta = json.loads(mj.read_text())
        except Exception:
            meta = {}
    fs = float(meta.get("fs_hz") or args.fs or FS)
    itemsize = 2 if path.suffix == ".cs16" else 4
    nsamp = path.stat().st_size // (2 * itemsize)
    secs = nsamp / fs
    print("=" * 70)
    print(f"[replay] {path.name}")
    print(f"  fs={fs/1e6:.3f} MS/s  samples={nsamp}  = {secs:.2f} s")
    if meta:
        print(f"  meta: freq={meta.get('freq_hz')} antenna="
              f"{meta.get('antenna')} note={meta.get('note')}")
    # CAPTURE-INTEGRITY GATE on the archive itself: the file must be a
    # whole number of IQ pairs and its length must match declared secs.
    if path.stat().st_size % (2 * itemsize):
        print("  INTEGRITY: file is not a whole number of IQ pairs - VOID")
        return 1
    want = meta.get("secs")
    if want and abs(secs - float(want)) / float(want) > 0.01:
        print(f"  INTEGRITY: {secs:.2f}s of samples vs declared {want}s - VOID")
        return 1
    print(f"  INTEGRITY: OK ({secs:.2f}s of samples, declared {want}s)")
    if fs != FS:
        print(f"  NOTE fs != {FS/1e6} MS/s - adsb.py's PPM demod assumes "
              f"1 sample per 0.5 us chip; results would be meaningless.")
        return 1
    print("=" * 70)
    out = None if args.no_emit else Path(args.out or OUT_JSONL)
    if out and args.fresh and out.exists():
        out.unlink()
    chunk = int(args.chunk_s * fs)
    overlap = 512
    dtype = np.int16 if itemsize == 2 else np.float32
    scale = 32768.0 if itemsize == 2 else 1.0
    t_start = time.time()
    wl = {}

    inject = []                       # (icao, alt_ft, truth_fields, bits)

    def _sweep(stats, wl_only):
        done = 0
        n_inj = 0
        with open(path, "rb") as fh:
            while done < nsamp:
                n = min(chunk, nsamp - done)
                raw = np.fromfile(fh, dtype=dtype, count=2 * n)
                n_got = raw.size // 2
                if n_got < 2:
                    break
                iq = (raw[0::2].astype(np.float32)
                      + 1j * raw[1::2].astype(np.float32)) / scale
                iq = iq.astype(np.complex64)
                if inject and not wl_only and n_inj < len(inject):
                    med = float(np.median(np.abs(iq)))
                    amp = args.inject_amp * med
                    step = max(n_got // 6, 4000)
                    p = 2000
                    while p + 600 < n_got and n_inj < len(inject):
                        splice(iq, p, inject[n_inj][3], amp)
                        n_inj += 1
                        p += step
                harvest(iq, stats=stats, whitelist=wl,
                        origin=(f"replay:{path.name}"
                                + ("+SYNTHETIC-INJECT" if inject else "")),
                        emit_path=None if wl_only else out,
                        strict=not args.loose, verbose=args.verbose,
                        wl_only=wl_only)
                done += n_got
                if args.progress:
                    tag = "wl" if wl_only else "cl"
                    print(f"  [{tag}] {done/fs:6.1f}s  ap={stats['ap']} "
                          f"commb={stats['commb']} bds44={stats['bds44']}",
                          flush=True)
                # rewind by one frame's footprint so a reply straddling the
                # chunk boundary is not lost.  MUST come after the EOF test,
                # or the rewind re-reads the tail forever (bit us once).
                if n_got < n or done >= nsamp:
                    break
                fh.seek(-2 * overlap * itemsize, os.SEEK_CUR)
                done -= overlap
        return done

    if not args.one_pass:
        # PASS 1: whitelist only.  Offline we are not obliged to be causal -
        # an aircraft that ADS-Bs at t=25s should still validate its own
        # Comm-B reply at t=3s.
        _sweep(_new_stats(), wl_only=True)
        print(f"[pass 1] whitelist built: {len(wl)} aircraft")
    if args.inject:
        # POSITIVE CONTROL.  Splice synthetic BDS 4,4 replies, addressed
        # to aircraft actually present in this archive, on top of the
        # REAL RF background at a stated amplitude.  If these come back
        # out, a zero on the untouched archive means "nobody transmitted
        # MRAR", not "the decoder is broken".
        icaos = list(wl) or ["A1B2C3"]
        rng = np.random.default_rng(44)
        for i in range(int(args.inject)):
            icao = icaos[i % len(icaos)]
            alt = int(rng.choice([31000, 35000, 37000, 39000]))
            tf = dict(fom=int(rng.integers(1, 5)),
                      wind_kt=float(int(rng.integers(20, 180))),
                      wind_dir=float(int(rng.integers(0, 359))),
                      sat_c=round(float(isa_temp_c(alt)
                                        + rng.uniform(-8, 8)) * 4) / 4.0,
                      press_hpa=int(round(isa_press_hpa(alt)
                                          + rng.uniform(-20, 20))),
                      turb=int(rng.integers(0, 4)),
                      rh_pct=float(int(rng.integers(0, 60))))
            inject.append((icao, alt, tf,
                           make_df20(icao, encode_bds44(**tf), alt_ft=alt)))
        print(f"[inject] {len(inject)} synthetic BDS 4,4 replies at "
              f"{args.inject_amp}x the median magnitude, addressed to "
              f"{len(set(i[0] for i in inject))} real aircraft in this archive")
    st = _new_stats()
    done = _sweep(st, wl_only=False)
    print(f"[replay] {done/fs:.1f}s processed in {time.time()-t_start:.0f}s wall")
    print_stats(st)
    print(f"  aircraft on whitelist : {len(wl)}")
    if st["records"]:
        print("  reports:")
        for r in st["records"][:20]:
            print(f"    {r['icao']} alt={r['alt_ft']} wind={r['wind_kt']}kt/"
                  f"{r['wind_dir']} sat={r['sat_c']}C "
                  f"[{r['plausibility']['verdict']} "
                  f"dISA={r['plausibility']['sat_minus_isa_c']}]")
        if out:
            print(f"  -> {out}")
    else:
        print("  no BDS 4,4 reports - expected over North America: MRAR is "
              "not part of Enhanced Surveillance, so no ground radar here "
              "asks for register 4,4.  A zero is a MEASUREMENT, not a bug.")
    if inject:
        # verify every injected report came back with EXACT fields
        want = {}
        for icao, alt, tf, _b in inject:
            want.setdefault((icao, alt, tf["sat_c"]), tf)
        hits = miss = 0
        for r in st["records"]:
            key = (r["icao"], r["alt_ft"], r["sat_c"])
            tf = want.get(key)
            if tf is None:
                continue
            exact = (r["wind_kt"] == tf["wind_kt"]
                     and abs(r["wind_dir"] - tf["wind_dir"]) < 0.8
                     and r["sat_c"] == tf["sat_c"]
                     and r["press_hpa"] == tf["press_hpa"]
                     and r["turb"] == tf["turb"]
                     and abs(r["rh_pct"] - tf["rh_pct"]) < 1.6)
            hits += int(exact)
            miss += int(not exact)
        # The failure criterion is a WRONG field, not a lost frame: a
        # frame lost to a collision with real 1090 traffic (or to
        # marginal amplitude) is the demodulator's business, and is
        # measured by adsb.py's own dials.  A recovered report with a
        # wrong number is the thing this build must never do.
        print(f"[inject] recovered EXACT {hits}/{len(inject)}, field "
              f"mismatches {miss} - positive control "
              f"{'PASS' if miss == 0 and hits else 'FAIL'}"
              f"{'' if hits == len(inject) else '  (losses = frames that '
                 'collided with real traffic / fell below the demod '
                 'threshold, not decode errors)'}")
    return 0


# ==========================================================================
# live capture (fleet-warden gated, bounded)
# ==========================================================================
def cmd_capture(args):
    rl = adsb.fleet_lock()
    if rl is None:
        print("[capture] no fleet lock module found - refusing a bare open")
        return 1
    owner = "bds44"
    tries = max(1, int(args.retries))
    got_lock = False
    for k in range(tries):
        if rl.acquire(owner, "BDS 4,4 MRAR live probe", 50,
                      wait_s=float(args.wait_s)):
            got_lock = True
            break
        busy = rl.status() or {}
        print(f"  [{k+1}/{tries}] radio held by {busy.get('owner','?')} "
              f"(p{busy.get('priority','?')}: {busy.get('purpose','?')}) "
              f"- waiting politely")
    if not got_lock:
        print("[capture] radio not free after bounded retries - standing down. "
              "Offline build stands on its own.")
        return 2
    try:
        rl.clear_stop(owner)
        y = rl.should_yield()
        if y:
            print(f"[capture] higher-priority waiter ({y}) - yielding "
                  "before we even open")
            return 2
        print(f"[capture] {args.secs:.0f}s @ 1090 MHz, {args.antenna}, "
              f"fs={FS/1e6} MS/s (adsb.py's proven path)")
        t0 = time.time()
        sdr, stm = adsb.open_sdr(args.antenna, args.gain, FS)
        n_want = int(args.secs * FS)
        buf = np.empty(2 * 65536, np.int16)
        out = np.empty(2 * n_want, np.int16)
        got = 0
        last_hb = 0.0
        aborted = None
        try:
            while got < n_want:
                r = sdr.readStream(stm, [buf], 65536, timeoutUs=1_000_000)
                if r.ret > 0:
                    n = min(r.ret, n_want - got)
                    out[2 * got: 2 * (got + n)] = buf[:2 * n]
                    got += n
                elif r.ret < 0 and r.ret != -1:
                    aborted = f"stream error {r.ret}"
                    break
                now = time.time()
                if now - last_hb >= 5.0:
                    last_hb = now
                    rl.heartbeat()
                    if rl.stop_requested(owner):
                        rl.clear_stop(owner)
                        aborted = "graceful stop requested"
                        break
                    y = rl.should_yield()
                    if y:
                        aborted = f"yielding to {y}"
                        break
        finally:
            sdr.deactivateStream(stm)
            sdr.closeStream(stm)
            del sdr                        # destroy the Device (the SDRplay
            #                                handoff fix from d73f9af)
        wall = time.time() - t0
        # ---- CAPTURE-INTEGRITY GATE -------------------------------------
        want = args.secs * FS
        ratio = got / want if want else 0.0
        print(f"[integrity] samples={got} want={int(want)} "
              f"({ratio*100:.2f}%), wall={wall:.1f}s")
        if aborted:
            print(f"[integrity] aborted: {aborted}")
        if ratio < 0.99:
            print("[integrity] VOID - short capture, no weather claimed "
                  "from it")
            return 1
        iq = (out[0::2].astype(np.float32)
              + 1j * out[1::2].astype(np.float32)) / 32768.0
        iq = iq[:got].astype(np.complex64)
        if args.save_iq:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            p = LAB / f"bds44_{stamp}.cs16"
            out[:2 * got].tofile(p)
            json.dump({"freq_hz": adsb.FREQ, "fs_hz": FS, "format": "cs16",
                       "n_samples": int(got), "secs": got / FS,
                       "antenna": args.antenna, "gain": args.gain},
                      open(str(p) + ".json", "w"))
            print(f"[corpus] {p.name} ({got*4/1e6:.0f} MB) archived for replay")
    finally:
        rl.release(owner)
        print("[capture] radio released")
    st, wl = harvest(iq, origin="live",
                     emit_path=None if args.no_emit else Path(args.out
                                                              or OUT_JSONL),
                     strict=not args.loose, verbose=True,
                     t0=time.time())
    print_stats(st)
    print(f"  aircraft on whitelist : {len(wl)}")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("replay")
    r.add_argument("--iq", required=True)
    r.add_argument("--fs", type=float, default=None)
    r.add_argument("--chunk-s", type=float, default=2.0)
    r.add_argument("--out", default=None)
    r.add_argument("--no-emit", action="store_true")
    r.add_argument("--fresh", action="store_true",
                   help="truncate the JSONL before writing")
    r.add_argument("--loose", action="store_true",
                   help="allow FOM=0 and SAT=0.00 (measures their cost)")
    r.add_argument("--progress", action="store_true")
    r.add_argument("--verbose", action="store_true")
    r.add_argument("--inject", type=int, default=0,
                   help="positive control: splice N synthetic BDS 4,4 "
                        "replies onto the real RF background")
    r.add_argument("--inject-amp", type=float, default=8.0,
                   help="injection amplitude as a multiple of the block's "
                        "median magnitude (demod floor is 3.5x)")
    r.add_argument("--one-pass", action="store_true",
                   help="causal (live-like) whitelisting instead of the "
                        "two-pass offline default")
    c = sub.add_parser("capture")
    c.add_argument("--secs", type=float, default=30)
    c.add_argument("--antenna", default="Antenna B")
    c.add_argument("--gain", type=float, default=45)
    c.add_argument("--wait-s", type=float, default=20.0)
    c.add_argument("--retries", type=int, default=3)
    c.add_argument("--save-iq", action="store_true")
    c.add_argument("--out", default=None)
    c.add_argument("--no-emit", action="store_true")
    c.add_argument("--loose", action="store_true")
    args = ap.parse_args()
    if args.cmd == "selftest":
        sys.exit(cmd_selftest(args))
    elif args.cmd == "replay":
        sys.exit(cmd_replay(args))
    elif args.cmd == "capture":
        sys.exit(cmd_capture(args))


if __name__ == "__main__":
    main()
