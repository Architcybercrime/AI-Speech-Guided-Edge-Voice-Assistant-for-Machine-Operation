# Pin map and wiring - ESP32 DevKit V1 (classic ESP32)

Supersedes the earlier S3 map. Strapping pins 0/2/5/12/15 are avoided on purpose: a HIGH on
GPIO12 at boot (e.g. from the L298N ENA jumper) can stop the board from booting.

| Function | GPIO | Connects to |
|---|---|---|
| OLED SDA / SCL | 21 / 22 | OLED SDA / SCL (VCC to 3V3, GND to GND) |
| Relay IN1 (active LOW) | 32 | relay module IN1 |
| L298N IN1 / IN2 / ENA | 16 / 17 / 27 | L298N header (remove the ENA jumper) |
| Buzzer | 23 | buzzer long leg, other leg to GND |
| RGB LED R / G / B | 18 / 19 / 33 | each through 220 ohm; long (common) leg to GND |
| E-stop button | 4 | tactile button, diagonal legs: GPIO4 and GND |
| Reset button | 14 | tactile button, diagonal legs: GPIO14 and GND |
| Guard switch | 13 | jumper wire to GND = closed; pull it out = OPEN |
| INMP441 SCK / WS / SD | 26 / 25 / 34 | mic SCK / WS / SD (VDD 3V3, GND, L/R to GND) |

## Power
- Relay module VCC to ESP32 **5V** pin, GND to GND. Keep the yellow JD-VCC jumper on.
- 12 V adapter (+) to relay K1 **COM**; relay **NO** to L298N **+12V** terminal.
- 12 V adapter (-) to L298N **GND**; L298N GND also to an ESP32 GND (common ground).
- Motor to L298N OUT1 / OUT2. 100 uF across the relay module VCC/GND (stripe = GND).
- No L298N? Set `USE_L298N 0`, wire adapter(+) to COM, NO to motor(+), motor(-) to adapter(-).

## Demo-only deviation (state it honestly)
The tactile E-stop is normally-open. The real design uses a normally-closed latching button so
a cut wire also trips. The firmware latch and reset logic is the same.
