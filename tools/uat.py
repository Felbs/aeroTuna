"""uat.py - aeroTuna campaign: UAT 978 MHz (DO-282) - the OTHER ADS-B band.

Two treasures live on 978 MHz:
  * GA aircraft ADS-B (UAT downlink) - planes the 1090 scope never sees
  * FIS-B ground uplinks - the FAA literally BROADCASTS weather to
    cockpits: METARs, TAFs, NOTAMs, PIREPs, NEXRAD radar mosaics.
    Continuous, unencrypted, from ~600 ground towers. This decoder
    reads the same broadcast on your desk.

Physical layer: binary CPFSK at 1.041667 Mbps (+-312.5 kHz deviation),
sampled at 2.083334 MS/s = exactly 2 samples/bit. Frames start with a
36-bit sync word: 0xEACDDA4E2 for aircraft, its complement 0x153225B1D
for ground uplinks. FEC is Reed-Solomon over GF(2^8) poly 0x187, first
root alpha^120:
  uplink   : 6 byte-interleaved RS(92,72) blocks -> 432 payload bytes
  downlink : RS(30,18) basic / RS(48,34) long

FIS-B application layer: ground header (site position) + iterated info
frames -> APDUs -> product payloads; text products (id 413) are DLAC
6-bit packed METAR/TAF/PIREP/WINDS records.

LAW (sonde campaign): selftest proves MACHINERY, only live RF proves
CONVENTIONS - capture mode therefore archives every RS-clean payload to
lab/uat_uplinks.jsonl so parsing offsets can be iterated offline against
real frames without touching the radio again.

Modes:
  python uat.py selftest                  # no radio: RS + full IQ chain
  python uat.py capture --secs 30         # live 978 MHz first light
  python uat.py parse                     # re-parse archived payloads

Fleet citizenship: capture acquires the one-radio radio_lock (fleet_lock
via adsb.py) - never a bare open.
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
sys.path.insert(0, str(HERE))
import adsb  # noqa: E402  (fleet_lock + SDR plumbing live here)

LAB = HERE.parent / "lab"
LAB.mkdir(exist_ok=True)

FS = 2_083_334.0
BITRATE = 1_041_667.0
FREQ = 978e6
DEV_HZ = 312_500.0
SYNC_ADSB = 0xEACDDA4E2          # 36 bits
SYNC_UPLINK = 0x153225B1D        # bitwise complement of the above
UPLINK_BITS = 4416               # 552 bytes: 6 x RS(92,72) interleaved
ADSB_LONG_BITS = 384             # RS(48,34)
ADSB_SHORT_BITS = 240            # RS(30,18)

PRODUCT_NAMES = {
    8: "NOTAM", 9: "D-ATIS", 10: "TWIP", 11: "AIRMET", 12: "SIGMET",
    13: "SUA", 14: "G-AIRMET", 15: "CWA", 16: "NOTAM-TFR",
    63: "NEXRAD regional", 64: "NEXRAD CONUS",
    70: "Icing low", 71: "Icing high", 84: "Cloud tops",
    90: "Turbulence low", 91: "Turbulence high",
    103: "Lightning", 413: "Text METAR/TAF/PIREP/WINDS",
}


# ==========================================================================
# Reed-Solomon over GF(2^8), poly 0x187, fcr 120 (the UAT field)
# ==========================================================================
class RS:
    _EXP = None
    _LOG = None

    @classmethod
    def _tables(cls):
        if cls._EXP is None:
            exp = [0] * 512
            log = [0] * 256
            x = 1
            for i in range(255):
                exp[i] = x
                log[x] = i
                x <<= 1
                if x & 0x100:
                    x ^= 0x187
            for i in range(255, 512):
                exp[i] = exp[i - 255]
            cls._EXP, cls._LOG = exp, log
        return cls._EXP, cls._LOG

    def __init__(self, nroots, fcr=120):
        self.nroots = nroots
        self.fcr = fcr
        exp, log = self._tables()
        g = [1]
        for i in range(nroots):
            root = exp[(fcr + i) % 255]
            ng = [0] * (len(g) + 1)
            for j, c in enumerate(g):
                ng[j] ^= self._mul(c, root)
                ng[j + 1] ^= c
            g = ng
        self.gen = g[::-1]          # highest degree first

    @classmethod
    def _mul(cls, a, b):
        if a == 0 or b == 0:
            return 0
        exp, log = cls._tables()
        return exp[log[a] + log[b]]

    @classmethod
    def _div(cls, a, b):
        exp, log = cls._tables()
        return exp[(log[a] - log[b]) % 255]

    def encode(self, data):
        """Systematic: returns data + nroots parity bytes."""
        msg = list(data) + [0] * self.nroots
        for i in range(len(data)):
            c = msg[i]
            if c:
                for j in range(1, len(self.gen)):
                    msg[i + j] ^= self._mul(self.gen[j], c)
        return bytes(data) + bytes(msg[len(data):])

    @classmethod
    def _inv(cls, a):
        exp, log = cls._tables()
        return exp[(255 - log[a]) % 255]

    @classmethod
    def _gfpow(cls, x, p):
        exp, log = cls._tables()
        return exp[((log[x] * p) % 255 + 255) % 255]

    @classmethod
    def _poly_scale(cls, p, x):
        return [cls._mul(c, x) for c in p]

    @classmethod
    def _poly_add(cls, p, q):
        r = [0] * max(len(p), len(q))
        r[len(r) - len(p):] = p
        for i, c in enumerate(q):
            r[i + len(r) - len(q)] ^= c
        return r

    @classmethod
    def _poly_mul(cls, p, q):
        r = [0] * (len(p) + len(q) - 1)
        for j, qc in enumerate(q):
            if qc:
                for i, pc in enumerate(p):
                    r[i + j] ^= cls._mul(pc, qc)
        return r

    @classmethod
    def _poly_eval(cls, p, x):
        y = 0
        for c in p:
            y = cls._mul(y, x) ^ c
        return y

    def _syndromes(self, cw):
        exp, _ = self._tables()
        return [self._poly_eval(cw, exp[(i + self.fcr) % 255])
                for i in range(self.nroots)]

    def decode(self, codeword):
        """(corrected data bytes, n_errors) or (None, -1). The classic
        BM + Chien + Forney ladder (reedsolo formulation, fcr-aware)."""
        exp, log = self._tables()
        cw = list(codeword)
        n = len(cw)
        synd = self._syndromes(cw)
        if max(synd) == 0:
            return bytes(cw[:-self.nroots]), 0
        # Berlekamp-Massey error locator (high-degree-first list)
        err_loc = [1]
        old_loc = [1]
        for i in range(self.nroots):
            old_loc = old_loc + [0]
            delta = synd[i]
            for j in range(1, len(err_loc)):
                delta ^= self._mul(err_loc[-(j + 1)], synd[i - j])
            if delta != 0:
                if len(old_loc) > len(err_loc):
                    new_loc = self._poly_scale(old_loc, delta)
                    old_loc = self._poly_scale(err_loc, self._inv(delta))
                    err_loc = new_loc
                err_loc = self._poly_add(err_loc,
                                         self._poly_scale(old_loc, delta))
        while len(err_loc) > 1 and err_loc[0] == 0:
            err_loc = err_loc[1:]
        errs = len(err_loc) - 1
        if errs == 0 or errs * 2 > self.nroots:
            return None, -1
        # Chien search (locator evaluated LOW-degree-first, per reedsolo)
        loc_rev = err_loc[::-1]
        err_pos = [n - 1 - i for i in range(n)
                   if self._poly_eval(loc_rev, exp[i % 255]) == 0]
        if len(err_pos) != errs:
            return None, -1
        # Forney via errata locator/evaluator
        coef_pos = [n - 1 - p for p in err_pos]
        e_loc = [1]
        for p in coef_pos:
            e_loc = self._poly_mul(e_loc,
                                   self._poly_add([1], [exp[p % 255], 0]))
        rem = self._poly_mul(list(synd[::-1]), e_loc)
        err_eval = rem[-(errs + 1):][::-1]        # low-degree-first
        X = [exp[p % 255] for p in coef_pos]
        E = [0] * n
        for i, Xi in enumerate(X):
            Xi_inv = self._inv(Xi)
            prime = 1
            for j, Xj in enumerate(X):
                if j != i:
                    prime = self._mul(prime, 1 ^ self._mul(Xi_inv, Xj))
            if prime == 0:
                return None, -1
            y = self._poly_eval(err_eval[::-1], Xi_inv)
            # NB: -fcr (not reedsolo's 1-fcr): our evaluator keeps the
            # remainder one degree shifted; pinned by the 0x42 trace and
            # the 10-error capacity selftest
            y = self._mul(self._gfpow(Xi, -self.fcr), y)
            E[err_pos[i]] = self._div(y, prime)
        for k in range(n):
            cw[k] ^= E[k]
        if max(self._syndromes(cw)) != 0:
            return None, -1
        return bytes(cw[:-self.nroots]), errs


RS_UPLINK = RS(20)               # RS(92,72)
RS_SHORT = RS(12)                # RS(30,18)
RS_LONG = RS(14)                 # RS(48,34)


# ==========================================================================
# bits <-> bytes helpers
# ==========================================================================
def bits_to_bytes(bits):
    n = len(bits) // 8
    out = bytearray(n)
    for i in range(n):
        v = 0
        for j in range(8):
            v = (v << 1) | int(bits[8 * i + j])
        out[i] = v
    return bytes(out)


def bytes_to_bits(data):
    out = np.zeros(len(data) * 8, np.uint8)
    for i, b in enumerate(data):
        for j in range(8):
            out[8 * i + j] = (b >> (7 - j)) & 1
    return out


def deinterleave_uplink(raw552):
    """Transmitted byte k belongs to RS block k%6, symbol k//6."""
    blocks = []
    for i in range(6):
        blocks.append(bytes(raw552[j * 6 + i] for j in range(92)))
    return blocks


def interleave_uplink(blocks):
    out = bytearray(552)
    for i in range(6):
        for j in range(92):
            out[j * 6 + i] = blocks[i][j]
    return bytes(out)


# ==========================================================================
# DLAC 6-bit text (FIS-B text products)
# ==========================================================================
_DLAC = "\x03ABCDEFGHIJKLMNOPQRSTUVWXYZ\x1a\t\x1e\n| !\"#$%&'()*+,-./0123456789:;<=>?"


def dlac_decode(data):
    out = []
    acc = 0
    nb = 0
    for byte in data:
        acc = (acc << 8) | byte
        nb += 8
        while nb >= 6:
            nb -= 6
            out.append(_DLAC[(acc >> nb) & 0x3F])
    return "".join(out)


def dlac_encode(text):
    out = bytearray()
    acc = 0
    nb = 0
    for ch in text:
        acc = (acc << 6) | _DLAC.index(ch)
        nb += 6
        while nb >= 8:
            nb -= 8
            out.append((acc >> nb) & 0xFF)
    if nb:
        out.append((acc << (8 - nb)) & 0xFF)
    return bytes(out)


# ==========================================================================
# FIS-B application layer
# ==========================================================================
def parse_uplink_payload(p):
    """432 RS-clean bytes -> ground header + products. Field offsets follow
    the community-validated uat2text/dump978 layout; the archive in
    lab/uat_uplinks.jsonl exists so these can be re-iterated offline
    (LAW: only live frames prove conventions)."""
    lat_raw = (p[0] << 15) | (p[1] << 7) | (p[2] >> 1)
    lon_raw = ((p[2] & 1) << 23) | (p[3] << 15) | (p[4] << 7) | (p[5] >> 1)
    if lat_raw & (1 << 22):
        lat_raw -= 1 << 23
    if lon_raw & (1 << 23):
        lon_raw -= 1 << 24
    site = {"lat": round(lat_raw * 360.0 / (1 << 24), 4),
            "lon": round(lon_raw * 360.0 / (1 << 24), 4),
            "utc_coupled": bool(p[6] & 0x80),
            "app_data_valid": bool((p[6] >> 5) & 1),
            "slot_id": p[6] & 0x1F,
            "tisb_site_id": p[7] >> 4}
    products = []
    d = p[8:]
    off = 0
    while off + 2 <= len(d):
        ln = (d[off] << 1) | (d[off + 1] >> 7)
        ftype = d[off + 1] & 0x0F
        if ln == 0:
            break
        frame = bytes(d[off + 2: off + 2 + ln])
        off += 2 + ln
        if ftype != 0 or len(frame) < 4:
            continue
        pid = ((frame[0] & 0x1F) << 6) | (frame[1] >> 2)
        t_opt = frame[1] & 0x03
        hdr = {0: 4, 1: 5, 2: 6, 3: 8}.get(t_opt, 4)
        body = frame[hdr:]
        prod = {"id": pid, "name": PRODUCT_NAMES.get(pid, f"product {pid}"),
                "bytes": len(body)}
        if pid == 413:
            prod["text"] = [r.strip() for r in
                            dlac_decode(body).split("\x1e") if r.strip()]
        products.append(prod)
    return site, products


def build_uplink_payload(site_lat, site_lon, text):
    """Selftest encoder: one text APDU inside a valid 432-byte payload."""
    p = bytearray(432)
    lat_raw = int(round(site_lat / (360.0 / (1 << 24)))) & 0x7FFFFF
    lon_raw = int(round(site_lon / (360.0 / (1 << 24)))) & 0xFFFFFF
    p[0] = (lat_raw >> 15) & 0xFF
    p[1] = (lat_raw >> 7) & 0xFF
    p[2] = ((lat_raw << 1) & 0xFE) | ((lon_raw >> 23) & 1)
    p[3] = (lon_raw >> 15) & 0xFF
    p[4] = (lon_raw >> 7) & 0xFF
    p[5] = (lon_raw << 1) & 0xFE
    p[6] = 0x20                      # app data valid
    body = dlac_encode(text)
    apdu = bytearray(4)              # t_opt 0 header
    pid = 413
    apdu[0] = (pid >> 6) & 0x1F
    apdu[1] = ((pid & 0x3F) << 2)    # t_opt = 0
    frame = bytes(apdu) + body
    ln = len(frame)
    p[8] = (ln >> 1) & 0xFF
    p[9] = ((ln & 1) << 7) | 0x00    # frame type 0
    p[10:10 + ln] = frame
    return bytes(p)


# ==========================================================================
# modem
# ==========================================================================
def synth_iq(bits, snr_db=25.0, pad=2000, seed=1):
    """CPFSK: 1 = +DEV, 0 = -DEV, 2 samples/bit, phase-continuous."""
    rng = np.random.default_rng(seed)
    sym = np.repeat(np.where(np.asarray(bits) > 0, 1.0, -1.0), 2)
    dphase = 2.0 * np.pi * DEV_HZ / FS * sym
    phase = np.cumsum(dphase)
    iq = np.exp(1j * phase).astype(np.complex64)
    sig = np.concatenate([np.zeros(pad, np.complex64), iq,
                          np.zeros(pad, np.complex64)])
    amp = 10 ** (-snr_db / 20.0)
    sig += (rng.normal(0, amp, len(sig))
            + 1j * rng.normal(0, amp, len(sig))).astype(np.complex64)
    return sig


def sync_bits(word):
    return np.array([(word >> (35 - k)) & 1 for k in range(36)], np.uint8)


def _sync_pattern(word):
    chips = [1.0 if (word >> (35 - k)) & 1 else -1.0 for k in range(36)]
    return np.repeat(np.array(chips, np.float32), 2)


_PAT_UP = _sync_pattern(SYNC_UPLINK)
_PAT_DN = _sync_pattern(SYNC_ADSB)


def find_frames(iq, thresh=52.0):
    """FM-discriminate and correlate both sync words. Returns list of
    (kind, start_sample, soft_bits) with soft = 2-sample dphi sums."""
    dphi = np.angle(iq[1:] * np.conj(iq[:-1])).astype(np.float32)
    s = np.sign(dphi)
    out = []
    for kind, pat, nbits in (("uplink", _PAT_UP, UPLINK_BITS),
                             ("adsb", _PAT_DN, ADSB_LONG_BITS)):
        corr = np.correlate(s, pat, mode="valid")
        idx = np.where(corr > thresh)[0]
        last = -10
        for i in idx:
            if i - last < 5:            # cluster: keep first peak region max
                continue
            j = i + np.argmax(corr[i:i + 5])
            last = j
            a = j + 72
            b = a + 2 * nbits
            if b > len(dphi):
                continue
            pair = dphi[a:b].reshape(-1, 2).sum(axis=1)
            out.append((kind, int(j), pair))
    out.sort(key=lambda t: t[1])
    return out


def decode_uplink(soft):
    raw = bits_to_bytes((soft > 0).astype(np.uint8))
    blocks = deinterleave_uplink(raw)
    data = bytearray()
    nerr = 0
    for blk in blocks:
        d, e = RS_UPLINK.decode(blk)
        if d is None:
            return None, -1
        data += d
        nerr += e
    return bytes(data), nerr


def decode_adsb(soft):
    bits = (soft > 0).astype(np.uint8)
    raw = bits_to_bytes(bits)
    d, e = RS_LONG.decode(raw[:48])
    if d is not None:
        return d, e, "long"
    d, e = RS_SHORT.decode(raw[:30])
    if d is not None:
        return d, e, "short"
    return None, -1, None


# ==========================================================================
# commands
# ==========================================================================
def cmd_selftest(args):
    print("=" * 62)
    print("aeroTuna UAT 978 self-test")
    print("=" * 62)
    ok = True
    rng = np.random.default_rng(7)
    # 1. RS(92,72) correction to capacity
    print("[1] Reed-Solomon RS(92,72), 10-byte burst")
    data = bytes(rng.integers(0, 256, 72, dtype=np.uint8))
    cw = bytearray(RS_UPLINK.encode(data))
    for pos in rng.choice(92, 10, replace=False):
        cw[pos] ^= int(rng.integers(1, 256))
    dec, ne = RS_UPLINK.decode(bytes(cw))
    hit = dec == data and ne == 10
    print(f"    corrected {ne} errors, payload match: "
          f"{'OK' if hit else 'FAIL'}")
    ok &= hit
    # 2. downlink codes
    print("[2] RS(30,18) + RS(48,34)")
    for rs, k, n in ((RS_SHORT, 18, 30), (RS_LONG, 34, 48)):
        d0 = bytes(rng.integers(0, 256, k, dtype=np.uint8))
        cw = bytearray(rs.encode(d0))
        for pos in rng.choice(n, rs.nroots // 2, replace=False):
            cw[pos] ^= int(rng.integers(1, 256))
        dec, ne = rs.decode(bytes(cw))
        hit = dec == d0
        print(f"    RS({n},{k}): {ne} corrected  {'OK' if hit else 'FAIL'}")
        ok &= hit
    # 3. FIS-B text roundtrip (application layer alone)
    print("[3] FIS-B payload + DLAC text roundtrip")
    metar = "METAR KIAD 020252Z 00000KT 10SM CLR 24/17 A2992\x1e"
    p = build_uplink_payload(38.95, -77.45, metar)
    site, prods = parse_uplink_payload(p)
    hit = (abs(site["lat"] - 38.95) < 0.01 and abs(site["lon"] + 77.45) < 0.01
           and prods and prods[0]["id"] == 413
           and prods[0]["text"] == [metar.strip("\x1e").strip()])
    print(f"    site=({site['lat']},{site['lon']}) "
          f"products={[(x['id'], x.get('text')) for x in prods]}")
    print(f"    {'OK' if hit else 'FAIL'}")
    ok &= hit
    # 4. the whole radio: payload -> RS -> interleave -> CPFSK IQ -> demod
    print("[4] full synthetic IQ chain (uplink)")
    blocks = [RS_UPLINK.encode(p[i * 72:(i + 1) * 72]) for i in range(6)]
    raw = interleave_uplink(blocks)
    bits = np.concatenate([sync_bits(SYNC_UPLINK), bytes_to_bits(raw)])
    iq = synth_iq(bits, snr_db=12.0)
    frames = find_frames(iq)
    hit = False
    for kind, start, soft in frames:
        if kind != "uplink":
            continue
        payload, ne = decode_uplink(soft)
        if payload is None:
            continue
        site, prods = parse_uplink_payload(payload)
        if prods and prods[0].get("text") == [metar.strip("\x1e").strip()]:
            hit = True
            print(f"    sync at {start}, RS errors {ne}, text recovered: "
                  f"'{prods[0]['text'][0][:44]}...'")
            break
    print(f"    {'OK' if hit else 'FAIL'}")
    ok &= hit
    # 5. downlink frame through the same air
    print("[5] full synthetic IQ chain (ADS-B long)")
    d0 = bytes(rng.integers(0, 256, 34, dtype=np.uint8))
    cw = RS_LONG.encode(d0)
    bits = np.concatenate([sync_bits(SYNC_ADSB), bytes_to_bits(cw)])
    iq = synth_iq(bits, snr_db=12.0, seed=3)
    hit = False
    for kind, start, soft in find_frames(iq):
        if kind != "adsb":
            continue
        d, ne, ln = decode_adsb(soft)
        if d == d0:
            hit = True
            print(f"    sync at {start}, RS({ln}) clean  OK")
            break
    if not hit:
        print("    FAIL")
    ok &= hit
    print("=" * 62)
    print("SELFTEST", "PASS" if ok else "FAIL")
    print("=" * 62)
    return 0 if ok else 1


def cmd_capture(args):
    rl = adsb.fleet_lock()
    if rl and not rl.acquire("uat_rx", "UAT 978 first light",
                             int(args.prio), wait_s=args.wait):
        busy = rl.status() or {}
        print(f"[uat] radio held by {busy.get('owner', '?')} "
              f"(p{busy.get('priority', '?')}) - aborting, no bare open")
        return 2
    try:
        import SoapySDR
        from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
        SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
        last = None
        for att in range(12):       # API refuses opens 60-90 s post-close
            try:                     # (post-yield beat can run ~30-40 s)
                sdr = SoapySDR.Device("driver=sdrplay")
                break
            except Exception as e:
                last = e
                print(f"[uat] open attempt {att + 1}/12: {e}")
                time.sleep(8.0)
        else:
            raise RuntimeError(f"SDR open failed after 12 tries: {last}")
        sdr.setSampleRate(SOAPY_SDR_RX, 0, FS)
        fs_rb = sdr.getSampleRate(SOAPY_SDR_RX, 0)
        sdr.setFrequency(SOAPY_SDR_RX, 0, FREQ)
        try:
            sdr.setAntenna(SOAPY_SDR_RX, 0, args.antenna)
        except Exception:
            pass
        try:
            sdr.setGainMode(SOAPY_SDR_RX, 0, False)
            sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", max(20, 59 - int(args.gain)))
            sdr.writeSetting("rfgain_sel", "0")
        except Exception:
            pass
        print(f"[uat] fs readback {fs_rb:.0f} (want {FS:.0f}), "
              f"freq {sdr.getFrequency(SOAPY_SDR_RX, 0)/1e6:.3f} MHz, "
              f"antenna {sdr.getAntenna(SOAPY_SDR_RX, 0)}")
        st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16)
        sdr.activateStream(st)
        n_want = int(args.secs * FS)
        buf = np.empty(2 * 65536, np.int16)
        got = 0
        chunks = []
        t0 = time.time()
        last_progress = t0
        while got < n_want:
            r = sdr.readStream(st, [buf], 65536, timeoutUs=1_000_000)
            if r.ret > 0:
                chunks.append(buf[:2 * r.ret].copy())
                got += r.ret
                last_progress = time.time()
            elif r.ret == -4:
                continue
            elif r.ret < 0 and r.ret != -1:
                raise RuntimeError(f"readStream {r.ret}")
            if time.time() - last_progress > 20.0:
                raise RuntimeError("stream stalled")
        wall = time.time() - t0
        sdr.deactivateStream(st)
        sdr.closeStream(st)
        if got < 0.95 * n_want:              # capture-integrity law
            print(f"[uat] INTEGRITY FAIL {got}/{n_want} - discarding")
            return 1
        raw = np.concatenate(chunks)
        iq = ((raw[0::2].astype(np.float32)
               + 1j * raw[1::2].astype(np.float32)) / 32768.0
              ).astype(np.complex64)
        print(f"[uat] {got/FS:.1f}s captured ({wall:.1f}s wall), demod ...")
    finally:
        if rl:
            rl.release("uat_rx")
    frames = find_frames(iq, thresh=args.thresh)
    n_up = sum(1 for k, *_ in frames if k == "uplink")
    n_dn = sum(1 for k, *_ in frames if k == "adsb")
    print(f"[uat] sync hits: {n_up} uplink, {n_dn} adsb")
    uplinks = []
    n_adsb_ok = 0
    for kind, start, soft in frames:
        if kind == "uplink":
            payload, ne = decode_uplink(soft)
            if payload is not None:
                uplinks.append((start, ne, payload))
        else:
            d, ne, ln = decode_adsb(soft)
            if d is not None:
                n_adsb_ok += 1
    print(f"[uat] RS-CLEAN: {len(uplinks)} uplinks, {n_adsb_ok} adsb frames")
    if uplinks:
        with open(LAB / "uat_uplinks.jsonl", "a") as f:
            for start, ne, payload in uplinks:
                f.write(json.dumps({"t": time.time(), "rs_err": ne,
                                    "hex": payload.hex()}) + "\n")
        print(f"[uat] archived to lab/uat_uplinks.jsonl")
        for start, ne, payload in uplinks[:5]:
            site, prods = parse_uplink_payload(payload)
            print(f"  site ({site['lat']},{site['lon']}) slot "
                  f"{site['slot_id']}: "
                  + "; ".join(f"{p['name']}({p['bytes']}B)" for p in prods))
            for p in prods:
                for line in (p.get("text") or [])[:4]:
                    print(f"    | {line[:76]}")
    return 0


def cmd_parse(args):
    path = LAB / "uat_uplinks.jsonl"
    if not path.is_file():
        print("no archive yet - run capture first")
        return 1
    n = 0
    texts = 0
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        payload = bytes.fromhex(rec["hex"])
        site, prods = parse_uplink_payload(payload)
        n += 1
        print(f"[{n}] site ({site['lat']},{site['lon']}) "
              f"slot {site['slot_id']} rs_err {rec['rs_err']}")
        for p in prods:
            print(f"    {p['name']} ({p['bytes']} B)")
            for t in (p.get("text") or []):
                texts += 1
                print(f"      | {t[:76]}")
    print(f"\n{n} uplinks, {texts} text records")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    c = sub.add_parser("capture")
    c.add_argument("--secs", type=float, default=30)
    c.add_argument("--antenna", default="Antenna B")
    c.add_argument("--gain", type=float, default=45)
    c.add_argument("--wait", type=float, default=30)
    c.add_argument("--thresh", type=float, default=52,
                   help="sync correlation threshold of 72 (RS gates ghosts)")
    c.add_argument("--prio", type=int, default=60,
                   help="radio_lock priority (default 60 = user-driven)")
    sub.add_parser("parse")
    args = ap.parse_args()
    if args.cmd == "selftest":
        sys.exit(cmd_selftest(args))
    elif args.cmd == "capture":
        sys.exit(cmd_capture(args))
    elif args.cmd == "parse":
        sys.exit(cmd_parse(args))


if __name__ == "__main__":
    main()
