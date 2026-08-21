# aeroTuna ✈️📡

**Adaptive ADS-B decoding — the TV Tuna method pointed at 1090 MHz.**

**Project site:** [felbs.software](https://felbs.software) · **Contact:** [E@felbs.software](mailto:E@felbs.software)

Born 2026-07-17 from the [Software-TV-Tuner](https://github.com/Felbs/Software-TV-Tuner)
lineage (TV Tuna → [Radio Tuna](https://github.com/Felbs/gr-radiotuna) →
[wxTuna](https://github.com/Felbs/wxTuna) → aeroTuna). Same thesis, new sky:
every decoder secretly knows how well it's doing — close the loop on it.

## The idea
Every Mode S message carries a 24-bit CRC. Stock decoders (the dump1090
family) rescue corrupted messages by **blind bit-flipping** against that
CRC. aeroTuna demodulates with a **per-bit confidence plane** (the energy
margin between the two PPM chips — the SOVA idea from our ATSC decoder),
so rescue flips the *weakest* bits first. Smarter search, more recovered
messages from weak, distant aircraft.

**First-light numbers (indoor antenna, 20 s):** 30 CRC-valid messages raw,
**+81 recovered by confidence-guided rescue** — 2.7× more traffic than the
raw demod alone. 17 aircraft decoded, callsigns and all.

## Quickstart
```bash
git clone https://github.com/Felbs/aeroTuna.git
cd aeroTuna
python tools/adsb.py selftest            # proves the whole pipeline - no radio needed
python tools/adsb.py capture --secs 20   # live 1090 MHz (needs an SDR, see below)
python tools/aero_panel.py               # THE ATC SCOPE: http://127.0.0.1:8646
```
**Dependencies:** `numpy`, `numba`, and the `SoapySDR` python bindings +
a driver for your SDR. Easiest path on any OS is
[radioconda](https://github.com/ryanvolz/radioconda) (has all three);
on Debian/Ubuntu: `apt install python3-numpy python3-numba python3-soapysdr soapysdr-module-all`.

## Tools
| Command | What it does |
|---|---|
| `python tools/adsb.py selftest` | No radio needed: CRC-24 against published Mode S test vectors, field decode, synthetic-IQ roundtrip, and a marginal-bit rescue proof. |
| `python tools/adsb.py capture --secs 20` | Live 1090 MHz: demod → CRC gate → confidence rescue → plane table (ICAO, callsign, altitude, speed). |
| `python tools/adsb.py shootout` | Antenna A/B/C compared by decoded-message count — pick your 1090 MHz antenna empirically, not by folklore. |
| `python tools/aero_panel.py` | **The ATC scope**: a standing receiver + localhost radar display — blips, leader lines, trails, data blocks, flight strips. `--replay lab/x.cs16` runs it from a frozen capture, no radio needed. |
| `python tools/uat.py selftest` | **UAT 978 MHz (DO-282)**: no radio — Reed-Solomon (92/72, 30/18, 48/34) at error capacity, FIS-B application layer, DLAC text, and a full synthetic-IQ chain that recovers an injected METAR through the whole radio path. |
| `python tools/uat.py capture --secs 60` | Live 978 MHz: FIS-B ground uplinks (METARs/TAFs/NOTAMs/NEXRAD the FAA broadcasts to cockpits) + GA aircraft UAT ADS-B. RS-clean payloads archive to `lab/uat_uplinks.jsonl`; `parse` re-reads them offline. |
| `python tools/bds.py selftest` | **BDS 4,4 — airliners as radiosondes.** No radio: field roundtrip through the real frame path, status-bit rejection, whitelist discipline, and a Monte-Carlo false-positive measurement against the registers that actually fly (2,0 / 4,0 / 5,0 / 6,0). |
| `python tools/bds.py replay --iq <cs16>` | Archived 1090 IQ → DF20/21 census, whitelist pass rate, register census, BDS 4,4 reports to `lab/bds44.jsonl`. `--inject N` splices synthetic MRAR replies onto the real RF background as a positive control. |
| `python tools/bds.py capture --secs 30` | Live MRAR probe, fleet-warden gated (`radio_lock`, priority 50, polite waiting, heartbeats, capture-integrity gate). |

Hot loops are numba-jitted; a 20 s capture analyzes in ~3 s.

## The ATC scope
`aero_panel.py` serves a dark radar scope on `http://127.0.0.1:8646`
(localhost only — nothing leaves your machine, and no location is ever
configured: the view centers itself on the traffic it decodes).

Position comes from full **CPR decode**: local decode against each
aircraft's own last fix, global even/odd pairing to bootstrap — with two
honesty gates a naive decoder lacks:

* **two-pairing confirmation** — one global pairing never plots; a single
  miscorrected (rescued) frame can decode to a plausible fix hundreds of
  nm out, so a fix needs two independent pairings agreeing within 20 nm;
* **fleet-median range gate** — CPR boundary-straddle decodes
  self-consistently but a whole latitude zone (6°) off; 1090 MHz is
  line-of-sight (~240 nm), so bootstrap fixes >300 nm from the fleet
  median are rejected as mis-zones.

The panel wears its truth dials on screen: **delivery %** (samples
actually delivered vs wall×fs — a short read is data loss, shown, never
hidden), **rescued** (messages that exist only because of the confidence
plane), and an SDR state chip that names who holds the radio instead of
spinning silently.

**Two views:** the classic top-down scope, and a **3D perspective view**
(`V` key or the sidebar button) — orbit, pan and zoom a Google-Earth-style
camera around your receiver, aircraft drawn at (exaggerated) altitude on
stems above the map, dependency-free canvas projection. `[` `]` adjust the
altitude exaggeration.

**Basemap, offline by design:** coastlines, state borders and ~5,300
airports ship in the repo (`tools/basemap.json`, Natural Earth +
OurAirports, both public domain) and render under the scope — press
`M` to toggle. No tile servers, no network: your view of the sky never
leaves your machine. To pin the scope on your receiver ("RX" marker +
range/bearing per aircraft), drop a `lab/qth.json` with
`{"lat": ..., "lon": ...}` or use the **SET RX** fields in the sidebar —
either way it's written only to `lab/qth.json`, and `lab/` is gitignored,
so your position stays local by construction. Without it the scope
estimates your position from the traffic itself (hollow "RX est" marker —
decent, but biased tens of nm toward a directional antenna's beam).

## Listen to the sky (airband voice)
The scope doubles as an **airband receiver**: click any airport on the
map and its real published frequencies appear (tower, ground, approach,
ATIS — 14,000+ channels for 4,200 airports, embedded from OurAirports,
public domain). Click LISTEN and the radio retunes from 1090 MHz to the
channel, demodulates AM voice, and streams it to the browser — pilots,
controllers, ATIS weather loops. One tuner = one job: the scope pauses
while you listen (it says so), and STOP → SCOPE brings the planes back.
Squelch is adjustable (dB over adjacent-band noise, 0 = always open);
GUARD 121.5 is one click. VHF airband likes a wideband/discone antenna —
use the antenna selector. Machinery is field-verified (retune, SNR dial,
squelch, streaming); voice intelligibility depends on your antenna and
range — airborne transmitters are line-of-sight and much stronger than
ground stations.

## Status
- ✅ Demod + CRC + rescue validated (selftest) and proven live (17 aircraft first capture)
- ✅ Antenna shootout working — measure, don't assume (our indoor rabbit ears beat two bigger antennas at 1090)
- ✅ Confidence rescue beats blind flipping 1.33× at 0.0% hard-ghost rate on a frozen 480 MB corpus (`tools/rescue_ab.py`)
- ✅ CPR position decode (global + local, mode-s.org known vectors in selftest) + velocity vectors (track, vertical rate)
- ✅ Live ATC scope (`tools/aero_panel.py`)
- ✅ Mode S replies (DF11 + whitelist-gated DF4/5/20/21), airband voice + mini waterfall + band scan, UAT 978 decoder (selftest-complete; live reception pending)
- ✅ Miscorrection audit (`tools/miscorrect_audit.py`): confidence rescue repairs 42% more frames than blind flipping at a 1.6% field-mismatch rate (same failure class as blind); ghost-ICAO rate 3.5% → the panel requires 2 corroborating messages before an aircraft displays
- ✅ External referee (`tools/referee_pymodes.py`): pyModeS agrees with every decoded field on 279 native frames — zero mismatches (ICAO, callsign, altitude, speed, track, vertical rate)
- ✅ BDS 4,4 meteorological routine air report (`tools/bds.py`): full field set, and — the hard part — honest register discrimination. A Comm-B reply carries no register ID, so a wrong guess yields *plausible-looking garbage weather*. Measured: realistic BDS 2,0/4,0/5,0/6,0 payloads leak into "BDS 4,4" **0.0000%** of the time (n=20,000), because those registers put a status bit where 4,4 keeps its source code — a uniform-random payload leaks 0.12% at the parser and **0.0033%** after the standard-atmosphere referee. Positive control: synthetic MRAR spliced onto a real archive is recovered field-exact. **Live air 8/05, 90 s on Antenna B**: 24 aircraft, 4,317 parity-addressed replies, 967 DF20/21, of which **26 passed the ICAO whitelist** — and the discriminator named them: BDS 6,0 ×7, 5,0 ×4, 4,0 ×1, 14 unidentified, **BDS 4,4 ×0**. That is Enhanced Surveillance exactly as predicted. The zero is structural, not shyness: every one of those 26 replies carries MB bits 1-4 ≥ 5, which cannot be a BDS 4,4 source code, so *none* would parse as 4,4 even with every strictness knob off. MRAR appears not to be interrogated in this airspace.
- ⏳ Next: NEXRAD FIS-B renderer (needs UAT reception), surface position (TC 5-8), dump1090 IQ-level A/B

## Hardware
Any SoapySDR-supported SDR (reference: SDRplay RSPdx) + any antenna — the
shootout tells you which one. 1090 MHz loves short coax and line of sight.

## License
MIT
