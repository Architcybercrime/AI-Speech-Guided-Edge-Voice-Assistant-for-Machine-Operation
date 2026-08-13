# 00 — Scope & Frozen Vocabulary

**Status:** FROZEN as of Day 1. Any change requires a full model retrain and written justification.

## 1. System boundary

| In scope | Out of scope |
|---|---|
| Offline keyword spotting on ESP32-S3 | Cloud STT / internet connectivity |
| 1 wake word + 8 command keywords | Free-form speech, natural language parsing |
| Relay + PWM actuation of a single DC motor rig | Multi-machine coordination, fieldbus (Modbus/PROFINET) |
| Hardwired E-stop as authoritative safety input | Certified functional-safety rating (ISO 13849 PL) |
| English commands | Multi-language, code-switching |

## 2. Wake word

**`Hey Machina`**

Rationale:
- 4 syllables → low false-accept rate vs 2-syllable wake words
- Phonetically rare; does not collide with common shop-floor vocabulary
- No overlap with any command keyword

## 3. Command vocabulary (8 + 2 special classes)

| ID | Keyword | Action | Class | Confirm required |
|----|---------|--------|-------|------------------|
| 0 | `START`   | Motor ON at last set speed | motion | YES |
| 1 | `HALT`    | Motor OFF, state cleared | safe | no |
| 2 | `PAUSE`   | Motor OFF, state retained | safe | no |
| 3 | `RESUME`  | Motor ON at retained speed | motion | YES |
| 4 | `FASTER`  | Speed step +1 (max 5) | motion (bounded) | no |
| 5 | `SLOWER`  | Speed step -1 (min 1) | safe | no |
| 6 | `CONFIRM` | Approve pending motion command | control | n/a |
| 7 | `CANCEL`  | Discard pending command | control | n/a |
| 8 | `_unknown` | Any non-command speech → reject | negative class | n/a |
| 9 | `_silence` | Background / no speech | negative class | n/a |

### Design note — why not START / STOP

`START` and `STOP` are a minimal pair (differ by one phoneme in the coda). Under
factory noise these cross-confuse at a materially higher rate than acoustically
distant alternatives. `HALT` was selected as the stop keyword specifically to
maximise inter-class acoustic distance. This is a deliberate design decision,
not an arbitrary naming choice.

### Design note — negative classes are mandatory

Without `_unknown` and `_silence`, a softmax classifier assigns *every* input to
one of the 8 commands. These two classes are what make rejection possible.
`_unknown` must contain phonetically varied non-command speech, including
words that partially rhyme with the commands.

## 4. Safety architecture (summary — see 01-safety-architecture.md)

1. Voice is a **convenience input**, never a safety input.
2. The physical E-stop is **hardwired, normally-closed, latching**, and is always
   authoritative. Firmware cannot override it.
3. Any command causing motion requires a two-utterance handshake:
   `<command>` -> accept beep -> `CONFIRM` within 3 s -> actuate.
4. Fail-safe state = de-energised. Power loss, firmware hang (watchdog), or
   E-stop trip all result in motor OFF.
5. `START` / `RESUME` are rejected if the guard-door interlock reads open.

## 5. Success criteria

| Metric | Target |
|---|---|
| Wake-word recall @ 75 dBA ambient | > 95 % |
| Wake-word recall @ 90 dBA ambient | > 85 % |
| Command top-1 accuracy (held-out speakers) | > 95 % |
| False accepts per hour (10 h soak) | < 0.5 |
| End-to-end latency, utterance end -> relay (p95) | < 500 ms |
| Unsafe actuations (motion without confirm) | 0 |
| Continuous uptime, no reset | 72 h |

## 6. Dataset targets

| Item | Target |
|---|---|
| Speakers | >= 10 (mixed gender, mixed accent) |
| Reps per keyword per speaker | 20 |
| Real clips before augmentation | ~1600 command + ~600 unknown |
| Ambient noise recorded | >= 30 min |
| Post-augmentation training clips | ~15000 |
| Test split | Speaker-disjoint (never seen in training) |
