"""aero_panel.py - aeroTuna v2: the live ATC scope.

A localhost radar display fed by our own confidence-plane ADS-B decoder
(adsb.py). One process = one standing receiver + one tiny web panel:

  capture thread : holds the SDR (radio_lock citizenship when the fleet
                   lock module is present - priority 80, "human at the
                   scope"), streams 1090 MHz in half-second blocks,
                   demod -> CRC gate -> confidence rescue -> tracker.
  tracker        : per-aircraft state. Positions via CPR: local decode
                   against the aircraft's own last fix, global even/odd
                   pairing to bootstrap. Jump-rate sanity gate.
  http server    : 127.0.0.1:8646 - serves aero_scope.html + /state.json.

Honesty dials on the panel (the fleet laws, on screen):
  * delivery %  - samples actually delivered vs wall*fs (capture-integrity
                  law: a short read is data loss, show it, never hide it)
  * rescued     - how many shown messages exist only because of the
                  confidence plane (our reason to exist)
  * SDR state   - RUNNING / BUSY (who holds the radio) / YIELDED / REPLAY /
                  STALLED, never a silent spinner.

Modes:
  python aero_panel.py                          # live, Antenna A, port 8646
  python aero_panel.py --antenna "Antenna B" --gain 45
  python aero_panel.py --replay lab/adsb_x.cs16 --speed 2   # no SDR needed

The scope UI lives in aero_scope.html next to this file (read per request,
so UI edits are a browser refresh away).
"""
import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import adsb  # noqa: E402  (our decoder is the whole point)
try:    # optional: strict BDS 4,4 register discrimination (task #55).
    # Imported as bds_mod because `bds` is a local variable in the
    # tracker's Comm-B branch and would shadow the module.
    import bds as bds_mod  # noqa: E402
except Exception:
    bds_mod = None

FS = adsb.FS
BLOCK_S = 0.5
OWNER = "aero_panel"
MAX_RESCUE_PER_BLOCK = 300      # honesty cap: count what we skip


# ==========================================================================
# aircraft tracker
# ==========================================================================
class Tracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.df_census = {}
        self.ac = {}
        self.blocks = deque(maxlen=240)     # (t, cand, ok, rescued) ~2 min

    def _entry(self, icao, now):
        return self.ac.setdefault(icao, {
            "icao": icao, "first": now, "msgs": 0, "rescued": 0,
            "modes": 0, "cpr": {}, "trail": deque(maxlen=120)})

    def update(self, frames_ok, frames_rescued, df11, ap, n_candidates, now):
        with self.lock:
            for f, was_rescued in ([(f, False) for f in frames_ok]
                                   + [(f, True) for f in frames_rescued]):
                d = adsb.decode_fields(f["bits"])
                a = self._entry(d["icao"], now)
                a["msgs"] += 1
                a["rescued"] += int(was_rescued)
                a["last"] = now
                for k in ("callsign", "alt_ft", "speed_kt", "track_deg",
                          "vr_fpm"):
                    if k in d:
                        a[k] = d[k]
                if "lat_cpr" in d:
                    self._position(a, d, now)
            # DF11 all-calls: non-ADS-B aircraft announce themselves here
            n_modes = 0
            for f in df11:
                icao = f"{adsb._bf(f['bits'], 8, 32):06X}"
                a = self._entry(icao, now)
                a["msgs"] += 1
                a["modes"] += 1
                a["last"] = now
                n_modes += 1
            # DF4/5/20/21: parity-recovered address, whitelist-gated -
            # a 24-bit match against thin air would invent ghost planes
            for f in ap:
                a = self.ac.get(f["addr"])
                if a is None:
                    continue
                # census: which reply types actually reach us. US radars
                # often run elementary surveillance (DF4/5 only), so
                # Comm-B (DF20/21) harvests can be genuinely absent -
                # measure it instead of assuming (8/03).
                self.df_census[f["df"]] = self.df_census.get(f["df"], 0) + 1
                a["msgs"] += 1
                a["modes"] += 1
                a["last"] = now
                n_modes += 1
                for k, v in adsb.decode_modes_fields(
                        f["bits"], f["df"]).items():
                    if k == "alt_ft" and "lat" in a:
                        continue        # ES altitude wins for ES aircraft
                    a[k] = v
                # aircraft as weather probes: Comm-B registers, accepted
                # only when they agree with the plane's OWN ADS-B state
                if f["df"] in (20, 21):
                    bds, fields = adsb.classify_commb(f["bits"], a)
                    if bds:
                        met = a.setdefault("met", {})
                        met.update(fields)
                        met["t"] = now
                        met["bds"] = bds
                        if bds == "4,4":       # direct report wins
                            # ...but ONLY if the strict discriminator
                            # agrees the register really is 4,4 (task
                            # #55). A Comm-B reply carries no register
                            # ID, so a loose accept paints invented
                            # weather on the scope. Fail CLOSED: if the
                            # strict module is missing or throws, show
                            # nothing rather than something wrong.
                            conf = None
                            if bds_mod is not None:
                                try:
                                    conf = bds_mod.classify(
                                        f["bits"], truth=a,
                                        alt_ft=a.get("alt_ft"))
                                except Exception:
                                    conf = None
                            if conf and conf.get("bds") == "4,4":
                                g = conf["fields"]
                                a["wx"] = {"wind_kt": g.get("wind_kt"),
                                           "wind_dir": g.get("wind_dir"),
                                           "temp_c": g.get("sat_c"),
                                           "rh_pct": g.get("rh_pct"),
                                           "src": "BDS4,4",
                                           "verdict":
                                               conf["plausibility"]["verdict"],
                                           "t": now}
                        else:
                            d = adsb.derive_met(met)
                            if d:
                                d["src"] = "derived"
                                d["t"] = now
                                a["wx"] = d
            self.blocks.append((now, n_candidates, len(frames_ok),
                                len(frames_rescued), n_modes))
            # prune the long-gone
            for icao in [k for k, a in self.ac.items()
                         if now - a["last"] > 300.0]:
                del self.ac[icao]

    def _position(self, a, d, now):
        odd = d["cpr_odd"]
        a["cpr"][odd] = (d["lat_cpr"], d["lon_cpr"], now)
        fix = None
        if "lat" in a and now - a["pos_t"] < 180.0:
            lat, lon = adsb.cpr_local(d["lat_cpr"], d["lon_cpr"], odd,
                                      a["lat"], a["lon"])
            # jump-rate gate: reject a fix implying > ~1400 kt
            dt = max(now - a["pos_t"], 1.0)
            dnm = 60.0 * math.hypot(lat - a["lat"],
                                    (lon - a["lon"])
                                    * math.cos(math.radians(lat)))
            if dnm / dt * 3600.0 < 1400.0:
                fix = (lat, lon)
        if fix is None and 0 in a["cpr"] and 1 in a["cpr"]:
            e, o = a["cpr"][0], a["cpr"][1]
            if abs(e[2] - o[2]) <= 10.0:
                g = adsb.cpr_global(e[0], e[1], o[0], o[1],
                                    newest_odd=o[2] > e[2])
                # Confirmation gate: one global pairing NEVER plots. A single
                # miscorrected (rescued) or mispaired frame decodes to a
                # plausible fix hundreds of nm out - seen live on replay.
                # Two independent pairings that agree within 20 nm do.
                if g:
                    c = a.get("cand")
                    if c and now - c[2] < 30.0 and 60.0 * math.hypot(
                            g[0] - c[0], (g[1] - c[1])
                            * math.cos(math.radians(g[0]))) < 20.0:
                        fix = g
                    a["cand"] = (g[0], g[1], now)
            # Fleet-median range gate: CPR boundary-straddle decodes
            # self-consistently but a whole zone (6 deg lat) off - the
            # two-pairing gate can't see it. 1090 MHz line-of-sight is
            # ~240 nm, so a BOOTSTRAP fix > 300 nm from the median of
            # established fixes is a mis-zone, not an airplane. (No QTH
            # needed: the fleet centroid stands in for the receiver.)
            if fix:
                near = [(b["lat"], b["lon"]) for b in self.ac.values()
                        if b is not a and "lat" in b
                        and now - b["pos_t"] < 120.0]
                if len(near) >= 3:
                    mlat = sorted(p[0] for p in near)[len(near) // 2]
                    mlon = sorted(p[1] for p in near)[len(near) // 2]
                    dnm = 60.0 * math.hypot(
                        fix[0] - mlat,
                        (fix[1] - mlon) * math.cos(math.radians(mlat)))
                    if dnm > 300.0:
                        fix = None          # stays a candidate, never plots
        if fix:
            a["lat"], a["lon"] = fix
            a["pos_t"] = now
            tr = a["trail"]
            if not tr or 60.0 * math.hypot(fix[0] - tr[-1][0],
                                           fix[1] - tr[-1][1]) > 0.05:
                tr.append((round(fix[0], 5), round(fix[1], 5)))

    def snapshot(self, now):
        with self.lock:
            cut = now - 60.0
            cand = ok = resc = modes = 0
            for t, c, o, r, m in self.blocks:
                if t >= cut:
                    cand += c
                    ok += o
                    resc += r
                    modes += m
            out = []
            for a in self.ac.values():
                # corroboration gate (miscorrect_audit finding: ~3.5% of
                # rescued frames carry a ghost ICAO): an aircraft earns a
                # strip with 2+ messages - one lone frame never displays
                if a["msgs"] < 2:
                    continue
                e = {k: a[k] for k in ("icao", "msgs", "rescued", "modes",
                                       "callsign", "alt_ft", "speed_kt",
                                       "track_deg", "vr_fpm", "squawk",
                                       "lat", "lon", "wx") if k in a}
                e["age"] = round(now - a["last"], 1)
                if "pos_t" in a:
                    e["pos_age"] = round(now - a["pos_t"], 1)
                e["trail"] = list(a["trail"])[-40:]
                e["_first"] = a["first"]
                out.append(e)
            # stable order like a real strip bay - rows must not reshuffle
            out.sort(key=lambda e: e.pop("_first"))
            return {"aircraft": out,
                    "stats": {"cand_60": cand, "crc_ok_60": ok,
                              "rescued_60": resc, "modes_60": modes,
                              "msg_min": (ok + resc + modes),
                              "aircraft": len(out),
                              "with_pos": sum(1 for e in out if "lat" in e),
                              "with_wx": sum(1 for e in out if "wx" in e),
                              "df_census": dict(self.df_census)}}


# ==========================================================================
# airband AM voice (118-137 MHz: pilots, towers, ATIS)
# ==========================================================================
class AirbandDemod:
    """2 MS/s IQ (tuned OFFSET low, classic DC-spike dodge) -> 20 kHz mono
    PCM. Envelope AM with a min-tracking squelch floor: ATIS loops hold the
    channel open; an idle tower squelches to silence between calls."""
    RATE = 20000
    OFFSET = 200_000.0

    def __init__(self):
        def lp(ntaps, cut, fs):
            n = np.arange(ntaps) - (ntaps - 1) / 2
            h = np.sinc(2 * cut / fs * n) * np.hanning(ntaps)
            return (h / h.sum()).astype(np.float32)
        self.t1 = lp(101, 80e3, 2e6)       # decim 10 -> 200 kS/s
        self.t2 = lp(101, 3.5e3, 200e3)    # decim 10 -> 20 kS/s; airband
                                           # voice is <3 kHz - every extra
                                           # kHz of BW is pure noise
        self.z1 = np.zeros(100, np.complex64)
        self.z2 = np.zeros(100, np.complex64)
        self.n0 = 0                         # mixer phase (period 10 exact)
        self.agc = 1.0
        self.sq_db = 8.0                    # squelch threshold (UI-settable)
        self._win = np.hanning(4096).astype(np.float32)

    def _stage(self, x, taps, z, r):
        buf = np.concatenate([z, x])
        return (np.convolve(buf, taps, "valid")[::r].astype(np.complex64),
                buf[-(len(taps) - 1):])

    def process(self, iq):
        n = np.arange(self.n0, self.n0 + len(iq))
        self.n0 = (self.n0 + len(iq)) % 10
        x = iq * np.exp(-2j * np.pi * 0.1 * n).astype(np.complex64)
        x1, self.z1 = self._stage(x, self.t1, self.z1, 10)
        x, self.z2 = self._stage(x1, self.t2, self.z2, 10)
        # SNR vs the ADJACENT spectrum, not vs the channel's own history:
        # an always-on ATIS never shows a carrier-off floor on-channel
        # (first squelch design read a strong broadcast as 0 dB forever).
        # Stage-1 gives +-100 kHz: channel = |f|<8 kHz, noise = 30-80 kHz.
        spec = np.abs(np.fft.fftshift(
            np.fft.fft(x1[:4096] * self._win))) ** 2
        fbin = 200e3 / 4096
        mid = 2048
        chan = spec[mid - int(5e3 / fbin): mid + int(5e3 / fbin)].sum()
        lo_a, hi_a = int(30e3 / fbin), int(80e3 / fbin)
        adj = np.concatenate([spec[mid - hi_a: mid - lo_a],
                              spec[mid + lo_a: mid + hi_a]])
        noise = float(np.median(adj)) * (10e3 / fbin)
        snr_db = 10.0 * math.log10(max(chan, 1e-12) / max(noise, 1e-12))
        env = np.abs(x)
        audio = env - float(env.mean())
        if snr_db < self.sq_db:             # squelch closed
            audio[:] = 0.0
        else:
            rms = float(np.sqrt((audio ** 2).mean())) or 1e-6
            self.agc = 0.9 * self.agc + 0.1 * min(0.25 / rms, 60.0)
            audio = np.clip(audio * self.agc, -0.95, 0.95)
        return (audio * 32767).astype(np.int16), round(snr_db, 1)


# ==========================================================================
# receiver (live SDR or corpus replay)
# ==========================================================================
class Receiver(threading.Thread):
    def __init__(self, tracker, antenna, gain, replay=None, speed=1.0,
                 use_lock=True, listen_antenna=None):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.antenna = antenna
        self.scope_antenna = antenna
        self.listen_antenna = listen_antenna or antenna
        self.gain = gain
        self.replay = replay
        self.speed = speed
        self.rl = adsb.fleet_lock() if (use_lock and not replay) else None
        self.state = "STARTING"
        self.detail = ""
        self.delivery = deque(maxlen=40)    # per-block delivered fraction
        self.rescue_skipped = 0
        self.pending = {}                   # antenna/gain changes from UI
        self.shutdown = False
        self.last_block = None              # wedge watchdog timestamp
        self._clock = time.time()           # replay runs on STREAM time
        self.mode = "scope"                 # scope (1090) | listen (airband)
        self.listen_mhz = None
        self.listen_snr = None
        self.demod = None
        self.audio = deque(maxlen=40)       # (seq, pcm bytes) ~20 s
        self.audio_seq = 0
        self.audio_lock = threading.Lock()
        self.scan = None                    # {"t", "results", "busy"}
        self.wf = deque(maxlen=150)         # waterfall rows while listening
        self.wf_id = 0
        self._wf_win = np.hanning(4096).astype(np.float32)

    def now(self):
        """Tracker timebase: wall clock live, data time in replay (a 20x
        replay must not make the jump/pairing gates 20x tighter)."""
        return self._clock if self.replay else time.time()

    # ---- shared block processing ----
    def _process(self, iq, now):
        self.last_block = time.time()
        frames = adsb.demod_frames(iq)
        es = [f for f in frames if f["kind"] == "es"]
        good = [f for f in es if f["crc_ok"]]
        rescued = []
        bad = [f for f in es if not f["crc_ok"]]
        if len(bad) > MAX_RESCUE_PER_BLOCK:
            self.rescue_skipped += len(bad) - MAX_RESCUE_PER_BLOCK
            bad = bad[:MAX_RESCUE_PER_BLOCK]
        for f in bad:
            b2, nf = adsb.rescue(f["bits"], f["conf"])
            if b2 is not None:
                rescued.append({**f, "bits": b2})
        df11 = [f for f in frames if f["kind"] == "df11" and f["crc_ok"]]
        ap = [f for f in frames if f["kind"] == "ap"]
        self.tracker.update(good, rescued, df11, ap, len(frames), now)

    # ---- replay rail ----
    def _run_replay(self):
        self.state = "REPLAY"
        raw = np.fromfile(self.replay, np.int16)
        iq_all = (raw[0::2].astype(np.float32)
                  + 1j * raw[1::2].astype(np.float32)) / 32768.0
        iq_all = iq_all.astype(np.complex64)
        n_blk = int(BLOCK_S * FS)
        self.detail = f"{Path(self.replay).name} ({len(iq_all)/FS:.0f}s loop)"
        while not self.shutdown:
            for i in range(0, len(iq_all) - n_blk, n_blk):
                if self.shutdown:
                    return
                t0 = time.time()
                self._clock += BLOCK_S
                self._process(iq_all[i:i + n_blk + 300], self._clock)
                self.delivery.append(1.0)
                budget = BLOCK_S / max(self.speed, 0.01)
                time.sleep(max(0.0, budget - (time.time() - t0)))

    # ---- live rail ----
    def _open(self):
        import SoapySDR
        from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CS16
        SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
        sdr = SoapySDR.Device("driver=sdrplay")
        sdr.setSampleRate(SOAPY_SDR_RX, 0, FS)
        sdr.setFrequency(SOAPY_SDR_RX, 0, adsb.FREQ)
        try:
            sdr.setAntenna(SOAPY_SDR_RX, 0, self.antenna)
            got = sdr.getAntenna(SOAPY_SDR_RX, 0)     # readback law
            if got != self.antenna:
                self.detail = f"antenna readback '{got}'"
        except Exception:
            pass
        try:
            sdr.setGainMode(SOAPY_SDR_RX, 0, False)
            sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", max(20, 59 - int(self.gain)))
            sdr.writeSetting("rfgain_sel", "0")
        except Exception:
            pass
        st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16)
        sdr.activateStream(st)
        return sdr, st

    def _close(self, sdr, st):
        try:
            sdr.deactivateStream(st)
            sdr.closeStream(st)
        except Exception:
            pass

    def _read_block(self, sdr, st, n_want, max_stall_s=20.0):
        buf = np.empty(2 * 65536, np.int16)
        out = np.empty(2 * n_want, np.int16)
        got = 0
        t0 = time.time()
        last_progress = t0
        while got < n_want:
            r = sdr.readStream(st, [buf], 65536, timeoutUs=1_000_000)
            if r.ret > 0:
                n = min(r.ret, n_want - got)
                out[2 * got: 2 * (got + n)] = buf[:2 * n]
                got += n
                last_progress = time.time()
            elif r.ret == -4:           # overflow: dropped samples, keep going
                continue
            elif r.ret < 0 and r.ret != -1:
                raise RuntimeError(f"readStream error {r.ret}")
            if time.time() - last_progress > max_stall_s:
                raise RuntimeError("stream stalled")
        wall = max(time.time() - t0, 1e-6)
        iq = (out[0::2].astype(np.float32)
              + 1j * out[1::2].astype(np.float32)) / 32768.0
        return iq.astype(np.complex64), min(1.0, n_want / (wall * FS))

    def _run_live(self):
        n_blk = int(BLOCK_S * FS)
        stalls = 0
        while not self.shutdown:
            if self.rl and self.rl.stop_requested(OWNER):
                # honor the stop-file in EVERY loop, not just mid-stream:
                # the ERROR/BUSY cycles used to eat it via clear_stop on
                # each re-acquire, making graceful stop impossible there
                self.rl.clear_stop(OWNER)
                self.rl.release(OWNER)
                self.state = "STOPPED"
                self.detail = "stop-file honored"
                return
            if self.rl:
                if not self.rl.acquire(OWNER, "live ATC scope", 80,
                                       wait_s=5.0):
                    busy = self.rl.status() or {}
                    self.state = "BUSY"
                    self.detail = (f"radio held by {busy.get('owner', '?')} "
                                   f"(p{busy.get('priority', '?')})")
                    time.sleep(10.0)
                    continue
                self.rl.clear_stop(OWNER)
            try:
                sdr, st = self._open()
            except Exception as e:
                # HOLD the lock through open retries. Releasing here is a
                # LIVELOCK against politely-parking consumers: the TV
                # panel's sweeper needs ~2 s to see our lock and free the
                # device, but by its next look the lock was gone again -
                # "no available RSP devices" forever (first live night).
                self.state = "ERROR"
                self.detail = f"SDR open failed: {e} - lock held, retrying"
                if self.rl:
                    self.rl.heartbeat()
                time.sleep(15.0)
                continue
            self.state = "RUNNING"
            self.detail = ""
            tail = np.zeros(0, np.complex64)
            last_hb = 0.0
            try:
                while not self.shutdown:
                    if self.pending:
                        self._apply_pending(sdr, st)
                    if self.rl:
                        if time.time() - last_hb > 5.0:
                            self.rl.heartbeat()
                            last_hb = time.time()
                        why = self.rl.should_yield()
                        if why:
                            self.state = "YIELDED"
                            self.detail = why
                            break
                        if self.rl.stop_requested(OWNER):
                            self.rl.clear_stop(OWNER)
                            self.state = "STOPPED"
                            self.detail = "stop-file honored"
                            self.shutdown = True
                            break
                    iq, frac = self._read_block(sdr, st, n_blk)
                    self.delivery.append(frac)
                    stalls = 0
                    if self.mode == "listen":
                        pcm, snr = self.demod.process(iq)
                        self.listen_snr = snr
                        with self.audio_lock:
                            self.audio_seq += 1
                            self.audio.append((self.audio_seq, pcm.tobytes()))
                        # waterfall row: the whole 2 MHz around the channel,
                        # free from the same block - a keyed-up transmitter
                        # anywhere nearby paints a stripe you can click
                        acc = np.zeros(4096)
                        for k in range(3):
                            seg = iq[k * 4096:(k + 1) * 4096]
                            acc += np.abs(np.fft.fftshift(
                                np.fft.fft(seg * self._wf_win))) ** 2
                        db = 10.0 * np.log10(acc / 3 + 1e-12)
                        db = db.reshape(1024, 4).max(axis=1)   # peak-pool
                        floor = float(np.median(db))
                        row = np.clip(db - floor, 0, 60).astype(np.uint8)
                        self.wf_id += 1
                        self.wf.append((self.wf_id, row.tobytes()))
                        self.last_block = time.time()
                        tail = np.zeros(0, np.complex64)
                        continue
                    blk = np.concatenate([tail, iq]) if len(tail) else iq
                    self._process(blk, time.time())
                    tail = iq[-300:].copy()
            except RuntimeError as e:
                stalls += 1
                self.state = "STALLED"
                self.detail = f"{e} (x{stalls})"
            finally:
                self._close(sdr, st)
                # DESTROY the Device, not just the stream: the SDRplay API
                # session belongs to this PROCESS until the object dies, and
                # the service reports 'no available RSP devices' to every
                # other process while it lives. Measured 8/02: tools waited
                # 96 s in vain while we merely closed the stream; our own
                # reopen worked instantly (same session). del = real yield.
                try:
                    del sdr, st
                except Exception:
                    pass
                if self.rl:
                    self.rl.release(OWNER)
            if self.state == "YIELDED":
                time.sleep(20.0)
            elif self.state == "STALLED":
                if stalls >= 3:
                    # A standing scope OUTLIVES a wedged radio: report
                    # ERROR honestly and keep retrying on a slow cadence
                    # (first live night: exiting here killed the panel
                    # while the API was merely mid-recovery).
                    self.state = "ERROR"
                    self.detail = ("stream stalled 3x - SDR wedged? "
                                   "retrying every 60s (ladder: "
                                   "Restart-Service SDRplayAPIService, "
                                   "then USB replug)")
                    stalls = 0
                    time.sleep(60.0)
                else:
                    time.sleep(10.0)

    def _do_scan(self, sdr, st):
        """Airband activity sweep: hop 118-137 in 1.8 MHz windows, FFT
        carrier census. ~35 s; answers 'what is worth clicking RIGHT NOW'.
        Born of the 8/02 lesson: every hand-picked channel was quiet while
        28 carriers boomed elsewhere in the band."""
        from SoapySDR import SOAPY_SDR_RX
        self.scan = {"busy": True, "t": time.time(), "results": []}
        found = []
        win = np.hanning(8192).astype(np.float32)
        try:                                 # sweep on the airband antenna
            sdr.setAntenna(SOAPY_SDR_RX, 0, self.listen_antenna)
        except Exception:
            pass
        for cf in np.arange(119.0, 137.0, 1.8):
            sdr.setFrequency(SOAPY_SDR_RX, 0, cf * 1e6)
            if self.rl:
                self.rl.heartbeat()
            try:
                iq, _ = self._read_block(sdr, st, int(1.5 * FS),
                                         max_stall_s=10.0)
            except RuntimeError:
                continue
            self.last_block = time.time()   # scan is alive, not wedged
            iq = iq[int(0.3 * FS):]         # settle
            nseg = len(iq) // 8192
            psd = np.zeros(8192)
            for k in range(nseg):
                psd += np.abs(np.fft.fftshift(
                    np.fft.fft(iq[k * 8192:(k + 1) * 8192] * win))) ** 2
            db = 10 * np.log10(psd / max(nseg, 1) + 1e-12)
            floor = float(np.median(db))
            freqs = cf + (np.arange(8192) - 4096) * (FS / 8192) / 1e6
            hot = np.where(db > floor + 8)[0]
            i = 0
            while i < len(hot):
                j = i
                while j + 1 < len(hot) and hot[j + 1] - hot[j] <= 4:
                    j += 1
                pk = hot[i:j + 1][np.argmax(db[hot[i:j + 1]])]
                fmhz = float(freqs[pk])
                if 118.0 < fmhz < 136.99 and abs(fmhz - cf) > 0.02:
                    found.append((round(fmhz, 3),
                                  round(float(db[pk] - floor), 1)))
                i = j + 1
        found.sort(key=lambda t: -t[1])
        seen = set()
        results = []
        for f, s in found:
            k = round(f * 40) / 40          # dedup to 25 kHz raster
            if k not in seen:
                seen.add(k)
                results.append({"mhz": f, "db": s})
        self.scan = {"busy": False, "t": time.time(),
                     "results": results[:20]}
        # restore whatever we were doing (antenna AND frequency)
        if self.mode == "listen" and self.listen_mhz:
            sdr.setFrequency(SOAPY_SDR_RX, 0,
                             self.listen_mhz * 1e6 - AirbandDemod.OFFSET)
        else:
            try:
                sdr.setAntenna(SOAPY_SDR_RX, 0, self.scope_antenna)
                self.antenna = sdr.getAntenna(SOAPY_SDR_RX, 0)
            except Exception:
                pass
            sdr.setFrequency(SOAPY_SDR_RX, 0, adsb.FREQ)

    def _apply_pending(self, sdr, st=None):
        from SoapySDR import SOAPY_SDR_RX
        p, self.pending = self.pending, {}
        try:
            if "scan" in p and st is not None:
                self._do_scan(sdr, st)
            if "listen" in p:               # airband voice: retune, same fs
                mhz = float(p["listen"])
                if self.listen_antenna != self.antenna:
                    try:                     # VHF wants the wideband port
                        sdr.setAntenna(SOAPY_SDR_RX, 0, self.listen_antenna)
                        self.antenna = sdr.getAntenna(SOAPY_SDR_RX, 0)
                    except Exception:
                        pass
                sdr.setFrequency(SOAPY_SDR_RX, 0,
                                 mhz * 1e6 - AirbandDemod.OFFSET)
                self.demod = AirbandDemod()
                if "squelch" in p:
                    self.demod.sq_db = float(p["squelch"])
                with self.audio_lock:
                    self.audio.clear()
                self.mode = "listen"
                self.listen_mhz = mhz
                self.listen_snr = None
            elif "squelch" in p and self.demod:
                self.demod.sq_db = float(p["squelch"])
            if "scope" in p:                # back to 1090 ADS-B
                if self.scope_antenna != self.antenna:
                    try:
                        sdr.setAntenna(SOAPY_SDR_RX, 0, self.scope_antenna)
                        self.antenna = sdr.getAntenna(SOAPY_SDR_RX, 0)
                    except Exception:
                        pass
                sdr.setFrequency(SOAPY_SDR_RX, 0, adsb.FREQ)
                self.mode = "scope"
                self.listen_mhz = None
            if "antenna" in p:
                sdr.setAntenna(SOAPY_SDR_RX, 0, p["antenna"])
                got = sdr.getAntenna(SOAPY_SDR_RX, 0)
                self.antenna = got            # readback = the truth
                if self.mode == "listen":     # selector edits current mode's
                    self.listen_antenna = got
                else:
                    self.scope_antenna = got
                self.detail = "" if got == p["antenna"] \
                    else f"antenna readback '{got}'"
            if "gain" in p:
                sdr.setGain(SOAPY_SDR_RX, 0, "IFGR",
                            max(20, 59 - int(p["gain"])))
                self.gain = int(p["gain"])
        except Exception as e:
            self.detail = f"retune failed: {e}"

    def run(self):
        # numba warmup BEFORE the stream exists: first _scan call compiles
        adsb.demod_frames(np.zeros(4000, np.complex64))
        if self.replay:
            self._run_replay()
        else:
            self._run_live()

    def status(self):
        d = self.delivery
        state, detail = self.state, self.detail
        # Wedge watchdog (8/01, seen live the first night): readStream can
        # hang at the C level inside the SDRplay driver - no Python stall
        # guard fires, the thread freezes mid-call, and "RUNNING" becomes a
        # lie. The panel can't unstick C, but it must never lie: no block
        # for 30 s while claiming RUNNING = WEDGED on the dial, with the
        # recovery ladder in the detail.
        if state == "RUNNING" and self.last_block \
                and time.time() - self.last_block > 30.0:
            state = "WEDGED"
            detail = (f"no samples for {time.time() - self.last_block:.0f}s "
                      "- driver hung: restart panel; if it repeats, "
                      "Restart-Service SDRplayAPIService, then USB replug")
        return {"state": state, "detail": detail,
                "antenna": self.antenna, "gain": self.gain,
                "lock": bool(self.rl),
                "mode": self.mode, "listen_mhz": self.listen_mhz,
                "listen_snr": self.listen_snr, "scan": self.scan,
                "rescue_skipped": self.rescue_skipped,
                "delivery": round(100.0 * sum(d) / len(d), 1) if d else None}


# ==========================================================================
# http server
# ==========================================================================
def load_qth():
    """Optional receiver position for the scope's RX marker and centering.
    PRIVATE BY CONSTRUCTION: lives in lab/qth.json - lab/ is gitignored,
    nothing here is served beyond 127.0.0.1, and the basemap is embedded
    (no tile servers: your position never leaves the machine)."""
    try:
        q = json.loads((HERE.parent / "lab" / "qth.json").read_text())
        return {"lat": float(q["lat"]), "lon": float(q["lon"])}
    except Exception:
        return None


def make_handler(tracker, rx):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body if isinstance(body, bytes)
                             else body.encode())

        def do_GET(self):
            from urllib.parse import urlparse, parse_qs
            u = urlparse(self.path)
            if u.path in ("/", "/index.html"):
                page = HERE / "aero_scope.html"
                if page.is_file():
                    self._send(200, page.read_bytes(), "text/html")
                else:
                    self._send(500, "aero_scope.html missing", "text/plain")
            elif u.path == "/state.json":
                now = rx.now()
                snap = tracker.snapshot(now)
                snap["t"] = now
                snap["sdr"] = rx.status()
                snap["qth"] = load_qth()
                self._send(200, json.dumps(snap))
            elif u.path == "/basemap.json":
                bm = HERE / "basemap.json"
                if bm.is_file():
                    self._send(200, bm.read_bytes())
                else:
                    self._send(404, "{}")
            elif u.path == "/set":
                q = parse_qs(u.query)
                if "antenna" in q:
                    rx.pending["antenna"] = q["antenna"][0]
                if "gain" in q:
                    rx.pending["gain"] = q["gain"][0]
                self._send(200, '{"ok": true}')
            elif u.path == "/listen":
                q = parse_qs(u.query)
                if "stop" in q:
                    rx.pending["scope"] = True
                    self._send(200, '{"ok": true}')
                else:
                    try:
                        mhz = float(q["mhz"][0])
                        assert 108.0 <= mhz <= 137.0     # airband only
                    except Exception:
                        self._send(400, '{"ok": false}')
                        return
                    rx.pending["listen"] = mhz
                    if "sq" in q:
                        rx.pending["squelch"] = q["sq"][0]
                    self._send(200, '{"ok": true}')
            elif u.path == "/scan":
                rx.pending["scan"] = True
                if rx.state != "RUNNING":
                    # no radio yet: say so instead of queuing silently
                    rx.scan = {"busy": True, "waiting": True,
                               "t": time.time(), "results": []}
                self._send(200, '{"ok": true}')
            elif u.path == "/waterfall.json":
                q = parse_qs(u.query)
                since = int(q.get("since", ["0"])[0])
                rows = [{"id": i, "v": list(b)} for i, b in rx.wf
                        if i > since]
                # the tuned center sits OFFSET high inside the capture
                # (we tune 200 kHz low); tell the client the true axis
                center = (rx.listen_mhz or 0) \
                    - AirbandDemod.OFFSET / 1e6
                self._send(200, json.dumps(
                    {"center_mhz": center, "span_mhz": FS / 1e6,
                     "mode": rx.mode, "rows": rows[-40:]}))
            elif u.path == "/freqs.json":
                fj = HERE / "airband_freqs.json"
                self._send(200, fj.read_bytes() if fj.is_file() else b"{}")
            elif u.path == "/audio.wav":
                import struct
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                rate = 20000
                self.wfile.write(
                    b"RIFF" + struct.pack("<I", 0x7FFFFFF0) + b"WAVE"
                    + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate,
                                            rate * 2, 2, 16)
                    + b"data" + struct.pack("<I", 0x7FFFFFC8))
                seen = rx.audio_seq
                try:
                    while rx.mode == "listen" and not rx.shutdown:
                        with rx.audio_lock:
                            fresh = [c for s, c in rx.audio if s > seen]
                            if fresh:
                                seen = rx.audio[-1][0]
                        for c in fresh:
                            self.wfile.write(c)
                        if fresh:
                            self.wfile.flush()
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionError, OSError):
                    pass
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path == "/set_qth":
                try:
                    n = int(self.headers.get("Content-Length", 0))
                    q = json.loads(self.rfile.read(n))
                    lat, lon = float(q["lat"]), float(q["lon"])
                    assert -90 <= lat <= 90 and -180 <= lon <= 180
                except Exception:
                    self._send(400, '{"ok": false}')
                    return
                # lab/ is gitignored: the position lives on THIS machine
                # only and is served only to 127.0.0.1
                lab = HERE.parent / "lab"
                lab.mkdir(exist_ok=True)
                (lab / "qth.json").write_text(
                    json.dumps({"lat": lat, "lon": lon}))
                self._send(200, '{"ok": true}')
            else:
                self._send(404, "not found", "text/plain")
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8646)
    ap.add_argument("--antenna", default="Antenna A",
                    help="1090 MHz shootout winner was A (rabbit ears)")
    ap.add_argument("--gain", type=float, default=45)
    ap.add_argument("--replay", default=None,
                    help="cs16 file: feed the scope from a frozen capture")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="replay pacing (2 = twice real time)")
    ap.add_argument("--no-lock", action="store_true",
                    help="skip fleet radio_lock even if present")
    ap.add_argument("--listen-antenna", default=None,
                    help="antenna for airband voice (default: same as scope;"
                         " a wideband/discone port usually wins at VHF)")
    args = ap.parse_args()

    tracker = Tracker()
    rx = Receiver(tracker, args.antenna, args.gain, replay=args.replay,
                  speed=args.speed, use_lock=not args.no_lock,
                  listen_antenna=args.listen_antenna)
    rx.start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port),
                              make_handler(tracker, rx))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    mode = f"REPLAY {args.replay}" if args.replay else \
        f"LIVE {args.antenna} (lock={'on' if rx.rl else 'off'})"
    print(f"[aero_panel] ATC scope on http://127.0.0.1:{args.port}  [{mode}]")
    # The panel LIVES AND DIES with its receiver: a stop-file (or crash)
    # must take the whole process down, not leave a zombie HTTP server
    # squatting the port serving stale state (8/02: two panels double-
    # bound 8646 via SO_REUSEADDR - the zombie answered half the polls).
    try:
        while rx.is_alive():
            rx.join(timeout=1.0)
    except KeyboardInterrupt:
        rx.shutdown = True
        rx.join(timeout=10.0)
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
