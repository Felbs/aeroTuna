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
                a["msgs"] += 1
                a["modes"] += 1
                a["last"] = now
                n_modes += 1
                for k, v in adsb.decode_modes_fields(
                        f["bits"], f["df"]).items():
                    if k == "alt_ft" and "lat" in a:
                        continue        # ES altitude wins for ES aircraft
                    a[k] = v
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
                e = {k: a[k] for k in ("icao", "msgs", "rescued", "modes",
                                       "callsign", "alt_ft", "speed_kt",
                                       "track_deg", "vr_fpm", "squawk",
                                       "lat", "lon") if k in a}
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
                              "with_pos": sum(1 for e in out if "lat" in e)}}


# ==========================================================================
# receiver (live SDR or corpus replay)
# ==========================================================================
class Receiver(threading.Thread):
    def __init__(self, tracker, antenna, gain, replay=None, speed=1.0,
                 use_lock=True):
        super().__init__(daemon=True)
        self.tracker = tracker
        self.antenna = antenna
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
                        self._apply_pending(sdr)
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
                    blk = np.concatenate([tail, iq]) if len(tail) else iq
                    self._process(blk, time.time())
                    tail = iq[-300:].copy()
            except RuntimeError as e:
                stalls += 1
                self.state = "STALLED"
                self.detail = f"{e} (x{stalls})"
            finally:
                self._close(sdr, st)
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

    def _apply_pending(self, sdr):
        from SoapySDR import SOAPY_SDR_RX
        p, self.pending = self.pending, {}
        try:
            if "antenna" in p:
                sdr.setAntenna(SOAPY_SDR_RX, 0, p["antenna"])
                got = sdr.getAntenna(SOAPY_SDR_RX, 0)
                self.antenna = got            # readback = the truth
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
    args = ap.parse_args()

    tracker = Tracker()
    rx = Receiver(tracker, args.antenna, args.gain, replay=args.replay,
                  speed=args.speed, use_lock=not args.no_lock)
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
