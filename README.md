# AI Speech-Guided Edge Voice Assistant for Machine Operation

Fully offline voice control for industrial machinery, running on an
ESP32-S3 edge node. No cloud, no network dependency, sub-second actuation.

## Status
Phase 0 — scope frozen, dataset tooling in place. Hardware pending.

## Structure
- `docs/` — scope, safety architecture, benchmarks
- `hardware/` — BOM, wiring diagram, pin map
- `tools/` — dataset recording, augmentation, evaluation scripts
- `models/` — wake word (.ppn) and KWS model artifacts
- `firmware/` — ESP32-S3 Arduino project

## Safety
Voice is a convenience input only. The physical E-stop is hardwired,
normally-closed and always authoritative. See `docs/00-scope.md`.
