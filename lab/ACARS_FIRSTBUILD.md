# ACARS FIRST BUILD — the airplane text-message mode (aeroTuna campaign 2)

*2026-08-01 — tools/acars.py: synth + decoder + ladder + warden'd live scan.
Built OFFLINE; no SDR was opened during this build.*

## What this is

VHF ACARS per ARINC 618: 2400-baud MSK audio (1200/2400 Hz tones around an
1800 Hz center) **amplitude-modulated** onto the RF carrier — AM, not FM.
Channels scanned: **131.550** (primary), 130.025, 129.125, 131.725 MHz.
Decoded messages (tail, flight id, label, text) append as JSONL for the SKY
PANEL so aircraft text can join the ADS-B layer.

## Epistemic status

| Claim | Status | Evidence |
|---|---|---|
| Framing (SYN SYN SOH, mode+addr7+ack+label2+bid+STX+text+ETX/ETB, CRC-16 refl. 0x8408 init 0, odd parity/char, LSB-first) | **PROVEN** | 7 CRC-clean real messages decoded from acarsdec's off-air `test.wav` (real tails F-GTAE, PH-BXR, G-DBCK, LN-DYY; flight ids AF7728, KL1681, BA031T, DY083J; a `#DFB` engine report) |
| MSK bit convention | **PROVEN, and the wikis are wrong** | see "the convention", below |
| Full IQ chain (AM @2.048M +25kHz -> envelope -> demod) | **PROVEN on synth** | selftest field-exact roundtrip |
| Live reception on our discone | **UNTESTED** | user runs the live command below |
| Decode floor | measured below (prediction written first) |

## The convention (the build's one real discovery)

Two published stories exist for ACARS MSK: "1 = 1200 Hz, 0 = 2400 Hz"
(sigidwiki et al.) and acarsdec's coherent quadrature demod. They are not
the same code. A first port that treated the data bits as plain quadrature
signs decoded its own synth perfectly (self-consistency proves nothing) and
decoded **zero** real recordings. The dropped line was acarsdec's

    if(ch->MskS&2) putbit(-vo, ch);   /* period-4  +,+,-,-  twiddle */

The true convention, stated three equivalent ways:

1. **Differential-by-tone: a 2400 Hz bit period HOLDS the previous bit, a
   1200 Hz period FLIPS it.** (So "1 = 1200 Hz" is wrong twice over.)
2. Data = quadrature signs of the MSK phase trajectory XOR a period-4
   `0,0,1,1` twiddle (the 1800 Hz center rotates 3pi/2 per bit through the
   2400-baud lattice; the twiddle is that rotation).
3. ARINC's "pre-key of all binary ones" **is** the steady 2400 Hz tone we
   measured on every real burst — consistent only with (1)/(2).

Receiver-side, the two physical ambiguities (bit-lattice parity, twiddle
phase — acarsdec's `MskS ^= 2` on inverted-SYN) are resolved by hunting the
sync in all four twiddle phases; CRC+parity make false accepts negligible.

## Verification rail (repeatable)

    python tools\acars.py selftest      # offline, field-exact, ~20 s
    python tools\acars.py wavcheck <path>\test.wav

`test.wav` is acarsdec's own 12500 Hz 4-channel off-air recording
(github.com/TLeconte/acarsdec — GPLv2, so it is referenced, not vendored).
Result 8/01: **7/7 channels' bursts CRC-clean, fields sane.** The selftest
also proved the honesty gate (3 bit-periods of dead air -> 0 decodes, no
false accept) and the coordinate-free jsonl hook shape.

## Decode-floor PREDICTION (written before the ladder ran)

SNR definition (in the code, printed by the ladder): total AM signal power
(carrier included — it is functional for envelope detection) over noise in
the 10 kHz channel; complex AWGN white across the 2.048 MHz capture.
Sidebands sit 5.8 dB below total at mod index 0.85, so
Eb/N0 = SNR + 10log10(10k/2400) - 5.8 = SNR + 0.4 dB.

Field-exact needs ~650 consecutive correct bits (P = (1-BER)^650), i.e.
BER <= ~1e-4 for a reliable rung -> coherent-MSK Eb/N0 ~ 8.4 dB, plus an
envelope-detection small-signal penalty (~1-2 dB at this CNR) and ~1 dB of
implementation loss (truncated half-cosine matched filter, PLL jitter).

**Prediction: 10/10-exact floor at SNR ~ 10 +/- 2 dB; a graceful-failure
band (partial seeds) from ~6-9 dB; essentially nothing below ~5 dB.**

## Ladder results (10 seeds/rung, lab/acars_ladder.csv)

Run 8/01, P_signal = -4.67 dBfs, frame = the 51-char selftest downlink
(~650 payload bits):

| SNR (dB) | field-exact | note |
|---|---|---|
| clean, 20, 15, 12, **10** | **10/10** | **10 dB = the floor** |
| 9 | 8/10 | graceful edge |
| 8 | 2/10 | |
| 7 .. 3 | 0/10 | dead |

Prediction scored: floor landed exactly on the 10 dB center of the
"10 +/- 2 dB" call; the failure band (8-9 dB) is NARROWER than the
predicted 6-9 — the all-or-nothing CRC cliff is sharper than the
per-bit arithmetic suggested. Below the floor the decoder returns
**nothing** rather than garbage: across all 121 rungs, every message
that passed the CRC gate was field-exact (zero false accepts) — the
truth-dial property this lab builds decoders around.

Wobble note (7/29 measurement law): rungs use fixed seeds; N=10 >= 3.
The 2/10 at 8 dB is a rate on 10 draws, not a promise.

## Live command (for the user — agents do not run this)

    cd Z:\src\aeroTuna\tools
    python acars.py live --secs 600 --dwell 15 --antenna "Antenna C"

Warden citizenship: owner `acars_rx` at priority 50 with holder.ok gate,
per-chunk heartbeat, should_yield, stop-file honor (`request_stop('acars_rx')`
to wind it down — never Stop-Process), bounded open-retry, 20 s stall guard
on every grab, and a capture-integrity gate (>=95% of expected samples or
the chunk is discarded). Tunes 25 kHz LOW (DC-spur law), envelope-detects,
appends decodes to `Z:\src\skyTuna\data\acars.jsonl`:

    {"t":..., "id":"<tail>", "callsign":"<flight or tail>", "label":"H1",
     "text":"...", "channel_mhz":131.55, "src":"acars_rx", "comment":"..."}

`sky_panel.py` will need an `acars` entry in its LAYERS dict to display the
layer (deliberately not patched from this build); the record shape already
matches its Tail/TRACK_FIELDS conventions and carries no coordinates.

## Honest negatives / open edges

- The live path is built to the warden contract but has never touched the
  radio; gain (default IFGR mapping at 131 MHz) is a guess until first light.
- No error correction: CRC is a pure gate. acarsdec repairs 1-2 damaged
  characters via CRC syndromes; our ADS-B-style confidence rescue would fit
  naturally on the |vo| soft decisions. Deliberately absent from v1 so the
  ladder measures the honest floor.
- Envelope AM detection was chosen over coherent for offset-immunity; below
  ~6 dB SNR a coherent carrier-tracking front end would buy back some of the
  small-signal suppression. Untested lever.
- ETB multi-block reassembly not implemented (blocks print individually,
  `etb` flagged in the parse).
- Uplink messages (letter block-ids) decode but carry no flight id by
  design of the protocol; the sky panel join key for uplinks is tail only.
