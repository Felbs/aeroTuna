"""acars.py - aeroTuna campaign 2: ACARS, the airplane text-message mode.

VHF ACARS (ARINC 618 "Plain Old ACARS") is 2400-baud MSK on an audio
subcarrier, AMPLITUDE-modulated onto the RF carrier - AM, not FM. The
MSK rides at 1800 Hz center, deviating to 1200/2400 Hz tones; data bits
are the QUADRATURE SIGNS of the MSK phase trajectory (MSK-as-OQPSK, no
differential precoding) - verified 8/01 by porting acarsdec's demod
faithfully and decoding acarsdec's real off-air test.wav recordings with
this file (see `wavcheck`); the popular "1 = 1200 Hz" wiki story does
not survive contact with a real receiver. Framing per ARINC 618:
pre-key (steady 2400 Hz tone, measured on real bursts), bit sync '+''*',
char sync SYN SYN (0x16), SOH, then mode + 7-char address + tech-ack +
2-char label + block id + STX + text + ETX/ETB, CRC-16 (reflected
0x8408, init 0, LSB-first bytes, remainder 0 over frame+CRC), DEL
suffix. Every character is 7-bit ASCII + ODD parity in the MSB, sent
LSB first (all conventions lifted from acarsdec's acars.c/msk.c and
proven against its recordings).

Frequencies (MHz): 131.550 (primary), 130.025, 129.125, 131.725.

Chain: IQ @ 2.048 MS/s tuned 25 kHz LOW (the DC-spur law) -> shift to
DC -> decimate to 12500 -> AM envelope -> MSK matched-filter demod
(acarsdec port: 1800 Hz NCO, half-cosine filter spanning 2 bit periods,
3pi/2 bit clock, PLL) -> sync hunt (both polarities) -> deframe ->
parity + CRC gate -> fields. Decoded messages append as JSONL for the
SKY PANEL (skyTuna data/acars.jsonl - the aircraft text layer).

Modes (OFFLINE first - selftest/ladder/wavcheck never touch an SDR):
  selftest  - synthesized legal frame, field-exact roundtrip + CRC gates
  ladder    - calibrated AWGN decode floor (SNR definition printed)
  wavcheck  - decode an audio WAV (e.g. acarsdec's test.wav) - the
              real-recording regression rail
  live      - round-robin scan of the 4 channels UNDER THE WARDEN
              (radio_lock citizenship: holder.ok gate, heartbeat,
              should_yield, stop_requested, bounded open-retry,
              max_stall_s grab, capture-integrity gate)

Examples:
  python acars.py selftest
  python acars.py ladder
  python acars.py wavcheck path\\to\\test.wav
  python acars.py live --secs 300 --antenna "Antenna C"
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

HERE = Path(__file__).resolve().parent
LAB = HERE.parent / "lab"
LAB.mkdir(exist_ok=True)
SKY_JSONL = Path(r"Z:\src\skyTuna\data\acars.jsonl")

FS = 2_048_000.0            # capture rate (never 250k - the 8/01 law)
OFFSET = 25_000.0           # tune LOW by this; shift back digitally
INT_FS = 12_500.0           # internal demod rate (acarsdec's INTRATE)
BAUD = 2400.0
AUDIO_FS = 48_000           # synth rate: exactly 20 samples/bit
SPB48 = 20
CHANNELS_MHZ = (131.550, 130.025, 129.125, 131.725)

SYN, SOH, STX, ETX, ETB, DEL, NAK = 0x16, 0x01, 0x02, 0x03, 0x17, 0x7F, 0x15

# ==========================================================================
# CRC-16 (reflected 0x8408, init 0 - the ARINC 618 BCS, acarsdec's table)
# ==========================================================================
_CRC_TAB = np.zeros(256, np.uint16)
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ 0x8408 if _c & 1 else _c >> 1
    _CRC_TAB[_i] = _c


def crc16(data, crc=0):
    """Update over bytes (parity bits INCLUDED, like the real BCS)."""
    for b in data:
        crc = (crc >> 8) ^ int(_CRC_TAB[(crc ^ b) & 0xFF])
    return crc


def odd_parity(ch7):
    """7-bit char -> 8-bit wire char with odd parity in the MSB."""
    return ch7 | (0x80 if bin(ch7 & 0x7F).count("1") % 2 == 0 else 0)


# ==========================================================================
# Frame synthesis (ARINC 618 downlink/uplink block)
# ==========================================================================
def build_block(mode="2", addr="N402TU", ack="\x15", label="H1", bid="4",
                text="", etb=False):
    """The parity-protected byte block SOH..suffix + CRC + DEL.

    Returns (txt_bytes, full_bytes): txt_bytes = mode..ETX inclusive (what
    the CRC covers), full_bytes = SYN SYN SOH txt CRC0 CRC1 DEL DEL."""
    a = ("." * (7 - len(addr)) + addr)[:7]          # right-justified, dot pad
    body7 = mode + a + ack + label[:2].ljust(2) + bid
    txt = [odd_parity(ord(c)) for c in body7]
    if text:
        txt.append(odd_parity(STX))
        txt += [odd_parity(ord(c) & 0x7F) for c in text]
    txt.append(odd_parity(ETB if etb else ETX))
    crc = crc16(txt)
    full = [SYN, SYN, SOH] + txt + [crc & 0xFF, crc >> 8, DEL, DEL]
    return txt, full


def bytes_to_bits(byts):
    """Wire order: each byte LSB first."""
    out = np.zeros(len(byts) * 8, np.uint8)
    for i, b in enumerate(byts):
        for j in range(8):
            out[8 * i + j] = (b >> j) & 1
    return out


# ==========================================================================
# MSK modulator - the convention PROVEN against acarsdec's recordings:
# data bits are the quadrature signs of the MSK phase trajectory with a
# period-4 (+,+,-,-) twiddle on top (acarsdec's `if(MskS&2) putbit(-vo)`
# - the line the first port dropped, and the wavcheck rail caught).
# Equivalently, stated as tones: ACARS MSK is DIFFERENTIAL -
# a 2400 Hz bit period HOLDS the previous bit, a 1200 Hz period FLIPS it
# - which is exactly why the ARINC "pre-key of all ones" is the steady
# 2400 Hz tone we measured on real bursts. The popular "1 = 1200 Hz"
# wiki story is wrong on both counts.
# ==========================================================================
# Phase lives on the quarter-circle grid q*pi/2 and steps +-1 per bit
# (+1 <=> 2400 Hz, -1 <=> 1200 Hz). The receiver reads sign(cos) at even
# instants and sign(sin) at odd ones: readout(q) = 1 iff q mod 4 in
# {0, 1}; the wire bit is readout ^ twiddle4 where twiddle4 cycles
# 0,0,1,1. Given a target bit stream there is EXACTLY ONE reachable q at
# every step, so the trajectory is unique.
def _twiddle4(n, phase=0):
    return ((np.arange(n) + phase) % 4 >= 2).astype(np.uint8)


def bits_to_steps(bits):
    r = bits ^ _twiddle4(len(bits))          # target quadrature readouts
    q = 0 if r[0] else 2                     # seed satisfies readout 0
    steps = np.zeros(len(r) - 1, np.int8)
    for k in range(1, len(r)):
        up, dn = (q + 1) % 4, (q - 1) % 4
        want = 1 if r[k] else 0
        if (1 if up < 2 else 0) == want:
            steps[k - 1] = 1
            q = up
        else:
            steps[k - 1] = -1
            q = dn
    return steps


PREKEY_BITS = 200      # ~83 ms of all-ones = steady 2400 Hz, like real bursts


def frame_bits(full_bytes):
    """prekey + bit sync '+ *' + the framed bytes, as wire bits."""
    prekey = np.ones(PREKEY_BITS, np.uint8)
    return np.concatenate([prekey, bytes_to_bits([0x2B, 0x2A]),
                           bytes_to_bits(full_bytes)])


def msk_audio(bits, fs=AUDIO_FS):
    """bits -> constant-envelope MSK audio (float32, 1800 +- 600 Hz)."""
    steps = bits_to_steps(bits)
    freq = 1800.0 + 600.0 * np.concatenate([steps.astype(np.float64), [1.0]])
    spb = int(round(fs / BAUD))
    per_sample = np.repeat(freq, spb)
    phase = 2 * np.pi * np.cumsum(per_sample) / fs
    return np.sin(phase).astype(np.float32)


def am_iq(audio, fs_audio=AUDIO_FS, fs_iq=FS, offset=OFFSET, mod=0.85,
          amp=0.5, pad_s=0.15, noise=0.0, seed=0):
    """MSK audio -> AM onto an IQ carrier at +offset from DC @ fs_iq."""
    a = resample_poly(audio.astype(np.float64), 128, 3)     # 48k -> 2.048M
    n = np.arange(len(a), dtype=np.float64)
    iq = amp * (1.0 + mod * a) * np.exp(2j * np.pi * offset / fs_iq * n)
    rng = np.random.default_rng(seed)
    npad = int(pad_s * fs_iq)
    out = np.zeros(len(iq) + 2 * npad, np.complex128)
    out[npad:npad + len(iq)] = iq
    if noise > 0:
        out += noise * (rng.standard_normal(len(out))
                        + 1j * rng.standard_normal(len(out)))
    return out.astype(np.complex64)


# ==========================================================================
# Front end: IQ -> AM envelope audio at INT_FS
# ==========================================================================
def iq_to_audio(iq, fs=FS, offset=OFFSET):
    """Shift the 25 kHz-offset channel to DC, decimate to 12500 complex,
    envelope-detect (AM - offset- and phase-immune), remove DC."""
    n = np.arange(len(iq), dtype=np.float64)
    x = iq * np.exp(-2j * np.pi * offset / fs * n)
    x = resample_poly(x, 25, 4096)                  # 2.048M -> 12500 exact
    env = np.abs(x).astype(np.float64)
    # AC-couple like a real receiver: the burst's carrier pedestal is a
    # huge in-burst DC step that a global mean cannot remove (selftest
    # caught this on day one). One-pole DC blocker, fc ~ 100 Hz.
    from scipy.signal import lfilter
    return lfilter([1.0, -1.0], [1.0, -0.95], env)


# ==========================================================================
# MSK demodulator - faithful port of acarsdec's msk.c (proven on its
# real recordings via `wavcheck`), soft decisions out
# ==========================================================================
FLEN = int(INT_FS / 1200) + 1                       # 11: 2-bit-period filter
MFLTOVER = 12
_FLENO = FLEN * MFLTOVER + 1
_H = np.cos(2.0 * np.pi * 600.0 / INT_FS / MFLTOVER
            * (np.arange(_FLENO) - (_FLENO - 1) / 2))
_H[_H < 0] = 0.0
PLLG, PLLC = 38e-4, 0.52


def _msk_soft_impl(audio, h):
    """Per-sample port: NCO+mixer, matched filter, 3pi/2 bit clock, PLL.
    Returns (vo, lvl): alternating-quadrature soft bits and levels."""
    n_out = int(len(audio) * BAUD / INT_FS) + 8
    vo = np.zeros(n_out, np.float64)
    lvl = np.zeros(n_out, np.float64)
    inb = np.zeros(FLEN, np.complex128)
    p = 0.0
    df = 0.0
    clk = 0.0
    idx = 0
    ns = 0
    k = 0
    for i in range(len(audio)):
        s = 2.0 * np.pi * 1800.0 / INT_FS + df
        p += s
        if p >= 2.0 * np.pi:
            p -= 2.0 * np.pi
        inb[idx] = audio[i] * (math.cos(p) - 1j * math.sin(p))
        idx = (idx + 1) % FLEN
        clk += s
        if clk >= 3.0 * np.pi / 2.0 - s / 2.0:
            clk -= 3.0 * np.pi / 2.0
            o = int(MFLTOVER * (clk / s + 0.5))
            if o > MFLTOVER:
                o = MFLTOVER
            v = 0.0 + 0.0j
            for j in range(FLEN):
                v += h[o] * inb[(j + idx) % FLEN]
                o += MFLTOVER
            mag = abs(v)
            v /= mag + 1e-8
            if ns & 1:
                out = v.imag
                dphi = -v.real if out >= 0 else v.real
            else:
                out = v.real
                dphi = v.imag if out >= 0 else -v.imag
            if k < n_out:
                vo[k] = out
                lvl[k] = mag * mag / 4.0
                k += 1
            ns += 1
            df = PLLC * df + (1.0 - PLLC) * PLLG * dphi
    return vo[:k], lvl[:k]


if _HAVE_NUMBA:
    _msk_soft = njit(cache=True)(_msk_soft_impl)
else:
    _msk_soft = _msk_soft_impl


# ==========================================================================
# Deframe: sync hunt (both polarities) -> chars -> parity + CRC gate
# ==========================================================================
_SYNC = bytes(bytes_to_bits([SYN, SYN, SOH]))


def _char_at(bits, pos):
    v = 0
    for j in range(8):
        v |= int(bits[pos + j]) << j
    return v


def _deframe_one(bits):
    """One candidate hard-decision stream -> CRC-clean raw frames."""
    bs = bytes(bits)
    frames = []
    i = 0
    while True:
        pos = bs.find(_SYNC, i)
        if pos < 0:
            return frames
        j = pos + len(_SYNC)
        txt = []
        while j + 8 <= len(bs) and len(txt) < 260:
            c = _char_at(bits, j)
            j += 8
            txt.append(c)
            if c in (odd_parity(ETX), odd_parity(ETB)):
                break
        ok = (len(txt) >= 13 and j + 16 <= len(bs)
              and txt[-1] in (odd_parity(ETX), odd_parity(ETB)))
        if ok:
            c0, c1 = _char_at(bits, j), _char_at(bits, j + 8)
            perr = sum(1 for c in txt if bin(c).count("1") % 2 == 0)
            if crc16([c0, c1], crc16(txt)) == 0 and perr == 0:
                frames.append({"raw": txt, "bit_pos": pos})
                i = j + 16
                continue
        i = pos + 1


def deframe(bits):
    """Hard decisions -> frames, resolving the receiver's two physical
    ambiguities (bit-lattice parity and twiddle phase = acarsdec's
    MskS ^= 2 on ~SYN) by trying all four twiddle phases. A frame can
    only pass sync+parity+CRC under its true phase, so wrong phases
    contribute nothing; duplicates are keyed out by position."""
    frames, seen = [], set()
    for phase in range(4):
        for fr in _deframe_one(bits ^ _twiddle4(len(bits), phase)):
            key = (fr["bit_pos"], bytes(fr["raw"]))
            if key not in seen:
                seen.add(key)
                fr["twiddle"] = phase
                frames.append(fr)
    return frames


def parse_fields(raw):
    """Parity-stripped ARINC 618 block -> message dict (acars.c layout:
    mode, 7-char dot-padded address, ack, label, block id, STX, text)."""
    t = [c & 0x7F for c in raw]
    if len(t) < 13 or t[12] not in (STX, ETX):
        return None
    msg = {
        "mode": chr(t[0]),
        "tail": "".join(chr(c) for c in t[1:8] if c != ord(".")),
        "ack": "!" if t[8] == NAK else chr(t[8]),
        "label": "".join(chr(c) for c in t[9:11]).replace("\x7f", "d"),
        "bid": chr(t[11]),
        "downlink": chr(t[11]).isdigit(),
        "etb": t[-1] == ETB,
        "text": "",
    }
    body = t[13:-1] if t[12] == STX else []
    if msg["downlink"] and body:
        msg["no"] = "".join(chr(c) for c in body[:4])
        msg["fid"] = "".join(chr(c) for c in body[4:10])
        body = body[10:]
    msg["text"] = "".join(chr(c) for c in body)
    return msg


def demod_audio(audio_intfs):
    """AM-demodulated (or WAV) audio @ 12500 -> decoded message dicts."""
    vo, lvl = _msk_soft(np.ascontiguousarray(audio_intfs, np.float64), _H)
    bits = (vo > 0).astype(np.uint8)
    out = []
    for fr in deframe(bits):
        msg = parse_fields(fr["raw"])
        if msg is not None:
            sl = lvl[fr["bit_pos"]:fr["bit_pos"] + 8 * len(fr["raw"])]
            msg["lvl_db"] = round(10 * math.log10(sl.mean() + 1e-12), 1)
            out.append(msg)
    return out


def decode_iq(iq):
    return demod_audio(iq_to_audio(iq))


# ==========================================================================
# SKY PANEL hook: one JSON line per message (sky_panel.py Tail conventions:
# flat dict with "t" and "id"; no coordinates - ACARS has none to leak)
# ==========================================================================
def jsonl_record(msg, channel_mhz, t=None):
    return {
        "t": round(time.time() if t is None else t, 2),
        "id": msg["tail"] or "ACARS?",
        "callsign": msg.get("fid") or msg["tail"],
        "mode": msg["mode"], "label": msg["label"], "bid": msg["bid"],
        "text": msg["text"][:220],
        "channel_mhz": channel_mhz,
        "src": "acars_rx",
        "comment": f"ACARS {msg['label']}: {msg['text'][:80]}",
    }


def append_jsonl(recs, path=SKY_JSONL):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


# ==========================================================================
# selftest - the discipline: field-exact roundtrip before anything else
# ==========================================================================
TRUTH = dict(mode="2", tail="N402TU", ack="!", label="H1", bid="4",
             no="M01A", fid="TU0731",
             text="AEROTUNA ACARS SELFTEST. SKY PANEL TEXT LAYER RAIL.")


def synth_truth_iq(noise=0.01, seed=1, amp=0.5):
    _, full = build_block(mode=TRUTH["mode"], addr=TRUTH["tail"],
                          ack="\x15", label=TRUTH["label"], bid=TRUTH["bid"],
                          text=TRUTH["no"] + TRUTH["fid"] + TRUTH["text"])
    audio = msk_audio(frame_bits(full))
    return am_iq(audio, noise=noise, seed=seed, amp=amp)


def cmd_selftest(_args):
    print("=" * 66)
    print("aeroTuna ACARS self-test (no SDR touched)")
    print("=" * 66)
    ok = True
    print("[1] CRC-16 known vector ('123456789' -> 0x2189, CRC-16/KERMIT)")
    v = crc16(b"123456789")
    print(f"    got {v:#06x}  {'OK' if v == 0x2189 else 'FAIL'}")
    ok &= v == 0x2189
    print("[2] block CRC self-consistency (frame + CRC bytes -> 0)")
    txt, full = build_block(text="TEST")
    r = crc16(full[3:-2])
    print(f"    remainder={r}  {'OK' if r == 0 else 'FAIL'}")
    ok &= r == 0
    print("[3] field-exact roundtrip: frame -> MSK -> AM IQ @2.048M+25k "
          "-> envelope -> demod")
    msgs = decode_iq(synth_truth_iq())
    if len(msgs) == 1:
        m = msgs[0]
        fields = ("mode", "tail", "ack", "label", "bid", "no", "fid", "text")
        bad = [f for f in fields if m.get(f) != TRUTH[f]]
        print(f"    decoded 1 msg, lvl={m['lvl_db']} dB; "
              f"{'ALL ' + str(len(fields)) + ' FIELDS EXACT' if not bad else 'MISMATCH: ' + str(bad)}")
        ok &= not bad
    else:
        print(f"    FAIL: {len(msgs)} messages decoded (want 1)")
        ok = False
    print("[4] honesty gate: corrupted frame must NOT decode")
    iq = synth_truth_iq()
    k = len(iq) // 2
    iq[k:k + 2600] = 0                # ~3 bit periods of dead air mid-text
    n_bad = len(decode_iq(iq))
    print(f"    corrupted -> {n_bad} decodes  {'OK' if n_bad == 0 else 'FAIL'}")
    ok &= n_bad == 0
    print("[5] empty-text block (ETX directly after block id)")
    _, full5 = build_block(mode="2", addr="N402TU", ack="\x15", label="Q0",
                           bid="5", text="")
    m5 = demod_audio(resample_poly(
        msk_audio(frame_bits(full5)).astype(np.float64), 25, 96))
    ok5 = len(m5) == 1 and m5[0]["label"] == "Q0" and m5[0]["text"] == ""
    print(f"    {'OK' if ok5 else 'FAIL'} ({len(m5)} msg)")
    ok &= ok5
    print("[6] SKY PANEL jsonl hook shape")
    rec = jsonl_record(TRUTH | {"text": TRUTH["text"]}, 131.550)
    need = all(k in rec for k in ("t", "id", "callsign", "label", "text",
                                  "channel_mhz", "src"))
    no_pos = "lat" not in rec and "lon" not in rec
    print(f"    {json.dumps(rec)[:100]}...")
    print(f"    keys {'OK' if need else 'FAIL'}, coordinate-free "
          f"{'OK' if no_pos else 'FAIL'}")
    ok &= need and no_pos
    print("=" * 66)
    print("SELFTEST", "PASS - field-exact" if ok else "FAIL")
    print("=" * 66)
    return 0 if ok else 1


# ==========================================================================
# wavcheck - decode a real ACARS audio recording (acarsdec's test.wav)
# ==========================================================================
def cmd_wavcheck(args):
    import wave
    w = wave.open(args.wav)
    fr, nch, sw, n = (w.getframerate(), w.getnchannels(),
                      w.getsampwidth(), w.getnframes())
    print(f"[wavcheck] {args.wav}: {fr} Hz, {nch} ch, {n} frames")
    if sw != 2:
        print("[wavcheck] only 16-bit PCM supported")
        return 1
    raw = np.frombuffer(w.readframes(n), np.int16).reshape(-1, nch)
    total = 0
    for ch in range(nch):
        a = raw[:, ch].astype(np.float64) / 32768.0
        if fr != int(INT_FS):
            a = resample_poly(a, int(INT_FS), fr)
        a -= a.mean()
        msgs = demod_audio(a)
        total += len(msgs)
        for m in msgs:
            print(f"  ch{ch}: mode={m['mode']} tail={m['tail']:<7} "
                  f"ack={m['ack']} label={m['label']} bid={m['bid']} "
                  f"fid={m.get('fid', '-'):<7} no={m.get('no', '-'):<5} "
                  f"lvl={m['lvl_db']}dB")
            if m["text"]:
                print(f"        text: {m['text'][:100]!r}")
    print(f"[wavcheck] {total} CRC-clean message(s)")
    return 0 if total else 1


# ==========================================================================
# ladder - calibrated AWGN decode floor
# ==========================================================================
# SNR DEFINITION (stated precisely):
#   SNR = 10*log10( P_signal / P_noise_10k )
#   P_signal    = total power of the clean AM signal (carrier + MSK
#                 sidebands; the carrier is FUNCTIONAL for envelope
#                 detection so it counts - with mod index 0.85 the
#                 sidebands alone sit 5.8 dB below this number).
#   P_noise_10k = complex AWGN power falling inside +-5 kHz of the
#                 carrier = sigma^2 * (10 kHz / fs); noise is white
#                 across the 2.048 MHz capture band.
# This is an in-band CNR over the 10 kHz ACARS channel, directly
# comparable to ma3_ladder.py's convention (which excluded its carrier).
LADDER_CSV = LAB / "acars_ladder.csv"
LADDER_BW = 10e3


def run_rung(clean, p_sig, snr_db, seed, csv_path):
    import csv as _csv
    t0 = time.time()
    if snr_db is None:
        noisy, tag = clean, "clean"
    else:
        sigma2 = p_sig * FS / (LADDER_BW * 10 ** (snr_db / 10))
        rng = np.random.default_rng(seed)
        nz = (rng.standard_normal(len(clean))
              + 1j * rng.standard_normal(len(clean)))
        noisy = clean + (math.sqrt(sigma2 / 2) * nz).astype(np.complex64)
        tag = f"snr{snr_db:g}_s{seed}"
    msgs = decode_iq(noisy)
    exact = 0
    if len(msgs) == 1:
        m = msgs[0]
        exact = int(all(m.get(f) == TRUTH[f] for f in
                        ("mode", "tail", "ack", "label", "bid", "no",
                         "fid", "text")))
    row = dict(rung=tag, snr_db=("" if snr_db is None else snr_db),
               seed=seed, n_msgs=len(msgs), exact=exact,
               secs=round(time.time() - t0, 1))
    new = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[{tag}] msgs={len(msgs)} exact={exact} "
          f"{time.time()-t0:.1f}s", flush=True)
    return exact


def cmd_ladder(args):
    clean = synth_truth_iq(noise=0.0)
    p_sig = float(np.mean(np.abs(clean[clean != 0]) ** 2))
    print(f"[ladder] P_signal={10*math.log10(p_sig):+.2f} dBfs, "
          f"SNR = P_signal / noise-in-10kHz (carrier INCLUDED; "
          f"sidebands sit {10*math.log10(0.85**2/2/(1+0.85**2/2)):+.1f} dB "
          f"below)")
    rungs = args.rungs or ["clean", "20", "15", "12", "10", "9", "8",
                           "7", "6", "5", "4", "3"]
    for a in rungs:
        if a == "clean":
            run_rung(clean, p_sig, None, 0, LADDER_CSV)
            continue
        seeds = args.seeds
        if "x" in a:
            a, s = a.split("x")
            seeds = int(s)
        n_ok = 0
        for seed in range(1, seeds + 1):
            n_ok += run_rung(clean, p_sig, float(a), seed, LADDER_CSV)
        print(f"[ladder] SNR {a} dB: {n_ok}/{seeds} field-exact")
    return 0


# ==========================================================================
# live - 4-channel round-robin scan under the warden.  Never run by an
# agent: OFFLINE ONLY discipline; the user runs this.
# ==========================================================================
def _ensure_sdr_dll_path():
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


def _open_sdr(antenna, gain_db, retries=3):
    """Bounded open-retry; radio_lock must already be held by caller."""
    import SoapySDR
    from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
    SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
    last = None
    for att in range(1, retries + 1):
        try:
            sdr = SoapySDR.Device("driver=sdrplay")
            break
        except Exception as e:
            last = e
            print(f"[live] open attempt {att}/{retries} failed: {e}")
            time.sleep(5.0)
    else:
        raise RuntimeError(f"SDR open failed after {retries} tries: {last}")
    sdr.setSampleRate(SOAPY_SDR_RX, 0, FS)
    try:
        sdr.setAntenna(SOAPY_SDR_RX, 0, antenna)
        got_ant = sdr.getAntenna(SOAPY_SDR_RX, 0)     # readback law (7/31)
        if got_ant != antenna:
            print(f"[live] WARNING: antenna readback '{got_ant}' != "
                  f"requested '{antenna}'")
    except Exception as e:
        print(f"[live] antenna select: {e}")
    try:
        sdr.setGainMode(SOAPY_SDR_RX, 0, False)
        sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", max(20, 59 - int(gain_db)))
    except Exception as e:
        print(f"[live] gain: {e}")
    st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16)
    sdr.activateStream(st)
    return sdr, st


def _grab(sdr, st, n_want, max_stall_s=20.0):
    """Read n_want frames; a silent stream raises, never spins (8/01 law)."""
    buf = np.empty(2 * 65536, np.int16)
    out = np.empty(2 * n_want, np.int16)
    got = 0
    last_progress = time.time()
    while got < n_want:
        r = sdr.readStream(st, [buf], 65536, timeoutUs=1_000_000)
        if r.ret > 0:
            n = min(r.ret, n_want - got)
            out[2 * got: 2 * (got + n)] = buf[:2 * n]
            got += n
            last_progress = time.time()
        elif r.ret < 0 and r.ret != -1:
            raise RuntimeError(f"readStream error {r.ret}")
        if time.time() - last_progress > max_stall_s:
            raise RuntimeError(f"stream stalled >{max_stall_s:.0f}s "
                               f"at {got}/{n_want} samples")
    iq = (out[0::2].astype(np.float32)
          + 1j * out[1::2].astype(np.float32)) / 32768.0
    return iq.astype(np.complex64), got


def cmd_live(args):
    _ensure_sdr_dll_path()
    sys.path.insert(0, r"Z:\src\gr-radiotuna\tools")
    import radio_lock
    owner = "acars_rx"
    secs = min(float(args.secs), 3600.0)
    dwell = max(2.0, float(args.dwell))
    chans = CHANNELS_MHZ if not args.freq else tuple(args.freq)
    with radio_lock.Holder(owner, "ACARS 4-channel scan", 50,
                           wait_s=args.wait) as holder:
        if not holder.ok:                       # NEVER a bare open
            busy = radio_lock.status() or {}
            print(f"[live] warden acquire FAILED (held by "
                  f"{busy.get('owner', '?')} p{busy.get('priority', '?')}) "
                  f"- aborting, no bare open")
            return 2
        radio_lock.clear_stop(owner)            # stale stop-file guard
        print(f"[live] {secs:.0f}s scan of {chans} MHz on {args.antenna}, "
              f"dwell {dwell:.0f}s, tuned {OFFSET/1e3:.0f} kHz low "
              f"(owner={owner} p50)")
        sdr, st = _open_sdr(args.antenna, args.gain)
        n_msgs_total = 0
        t_end = time.time() + secs
        ci = 0
        try:
            from SoapySDR import SOAPY_SDR_RX
            while time.time() < t_end:
                why = radio_lock.should_yield()
                if why:
                    print(f"[live] yielding radio: {why}")
                    break
                if radio_lock.stop_requested(owner):
                    radio_lock.clear_stop(owner)
                    print("[live] stop-file honored - winding down")
                    break
                mhz = chans[ci % len(chans)]
                ci += 1
                sdr.setFrequency(SOAPY_SDR_RX, 0, mhz * 1e6 - OFFSET)
                _grab(sdr, st, int(0.25 * FS), max_stall_s=20.0)  # settle
                radio_lock.heartbeat()
                n_want = int(dwell * FS)
                t0 = time.time()
                iq, got = _grab(sdr, st, n_want, max_stall_s=20.0)
                wall = time.time() - t0
                radio_lock.heartbeat()
                if got < 0.95 * n_want or wall > 2.0 * dwell:
                    print(f"[live] capture-integrity FAIL on {mhz:.3f}: "
                          f"{got}/{n_want} in {wall:.1f}s - skipping chunk")
                    continue
                msgs = decode_iq(iq)
                if msgs:
                    recs = [jsonl_record(m, mhz) for m in msgs]
                    append_jsonl(recs, Path(args.jsonl))
                    n_msgs_total += len(msgs)
                    for m in msgs:
                        print(f"[{mhz:.3f}] {m['mode']} {m['tail']:<7} "
                              f"{m['label']} bid={m['bid']} "
                              f"fid={m.get('fid', '-')} lvl={m['lvl_db']}dB "
                              f"txt={m['text'][:60]!r}")
                else:
                    print(f"[{mhz:.3f}] no msg ({dwell:.0f}s)", flush=True)
        finally:
            try:
                sdr.deactivateStream(st)
                sdr.closeStream(st)
                del sdr
            except Exception:
                pass
    print(f"[live] done: {n_msgs_total} message(s) -> {args.jsonl}")
    return 0


# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    lad = sub.add_parser("ladder")
    lad.add_argument("rungs", nargs="*",
                     help="e.g. clean 12 8x10 (default full ladder)")
    lad.add_argument("--seeds", type=int, default=10)
    wv = sub.add_parser("wavcheck")
    wv.add_argument("wav", help="audio WAV (e.g. acarsdec's test.wav)")
    lv = sub.add_parser("live")
    lv.add_argument("--secs", type=float, default=300.0,
                    help="total scan time, capped 3600")
    lv.add_argument("--dwell", type=float, default=15.0,
                    help="seconds per channel visit")
    lv.add_argument("--antenna", default="Antenna C",
                    help="discone; Antenna B is a GPS patch now")
    lv.add_argument("--gain", type=float, default=40)
    lv.add_argument("--wait", type=float, default=60.0,
                    help="seconds to wait for the warden")
    lv.add_argument("--freq", type=float, action="append",
                    help="override channel MHz (repeatable)")
    lv.add_argument("--jsonl", default=str(SKY_JSONL))
    args = ap.parse_args()
    if args.cmd == "selftest":
        sys.exit(cmd_selftest(args))
    if args.cmd == "ladder":
        sys.exit(cmd_ladder(args))
    if args.cmd == "wavcheck":
        sys.exit(cmd_wavcheck(args))
    if args.cmd == "live":
        sys.exit(cmd_live(args))


if __name__ == "__main__":
    main()
