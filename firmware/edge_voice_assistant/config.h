// config.h — single source of truth for pins, thresholds and timings.
// Target: ESP32-S3-DevKitC-1 N16R8 (16 MB flash / 8 MB PSRAM).
//
// Reserved on N16R8, do not use: GPIO 19/20 (USB), 26-32 (flash),
// 33-37 (octal PSRAM). GPIO 0 and 45/46 are strapping pins.

#pragma once

// ---------------------------------------------------------------- AUDIO
// Two INMP441 share one I2S bus. Mic A: L/R -> GND (left channel).
// Mic B: L/R -> 3V3 (right channel). Both DOUT tie to the same SD line.
#define I2S_SCK_PIN        4    // BCLK
#define I2S_WS_PIN         5    // LRCL / word select
#define I2S_SD_PIN         6    // DOUT from mics -> ESP32 input

#define SAMPLE_RATE        16000
#define I2S_BITS           32   // INMP441 is 24-bit left-aligned in 32-bit slot
#define INMP441_SHIFT      11   // >> 11 to get int16. NOT 8 — classic bug.
#define AUDIO_CHANNELS     2    // stereo capture, averaged in software

#define WINDOW_MS          1000 // classifier input length
#define STRIDE_MS          250  // run inference every 250 ms
#define RING_BUFFER_MS     3000 // must be >= CMD_WINDOW_MS

// ---------------------------------------------------------------- RELAYS
// Most 5 V opto relay modules are ACTIVE LOW: digitalWrite(LOW) energises.
#define RELAY_ACTIVE_LOW   1
#define RELAY_1_PIN        10   // motor enable
#define RELAY_2_PIN        11   // direction
#define RELAY_3_PIN        12   // spare
#define RELAY_4_PIN        13   // spare

// ---------------------------------------------------------------- MOTOR (L298N)
#define MOTOR_IN1_PIN      16
#define MOTOR_IN2_PIN      17
#define MOTOR_ENA_PIN      18   // PWM speed
#define MOTOR_PWM_CH       0
#define MOTOR_PWM_FREQ     1000
#define MOTOR_PWM_BITS     8

#define SPEED_MIN          1
#define SPEED_MAX          5
#define SPEED_DEFAULT      3
// duty cycle per speed step (index 0 unused)
#define SPEED_DUTY_TABLE   {0, 90, 130, 165, 205, 255}

// ---------------------------------------------------------------- FEEDBACK
#define BUZZER_PIN         14
#define LED_READY_PIN      21   // green  - idle, armed
#define LED_LISTEN_PIN      1   // blue   - wake detected, listening
#define LED_FAULT_PIN       2   // red    - rejected / lockout
#define OLED_SDA_PIN        8
#define OLED_SCL_PIN        9
#define OLED_ADDR          0x3C

#define TONE_ACCEPT_HZ     1200 // rising  = accepted
#define TONE_REJECT_HZ      400 // falling = rejected
#define TONE_LOCKOUT_HZ     250 // triple  = lockout

// ---------------------------------------------------------------- SAFETY IN
// Both wired normally-closed to GND with INPUT_PULLUP.
// Reading LOW  = circuit intact = SAFE.
// Reading HIGH = circuit broken = TRIPPED (also catches a cut wire).
#define ESTOP_PIN          15
#define INTERLOCK_PIN       7   // guard door
#define BTN_RESET_PIN       3

// ---------------------------------------------------------------- DECISION
#define CONF_THRESHOLD     0.85f  // reject softmax below this
#define VOTE_N             3      // need N agreeing windows
#define VOTE_M             5      // out of the last M windows
#define CMD_WINDOW_MS      3000   // listening window opened by wake word
#define CONFIRM_TIMEOUT_MS 3000   // time allowed to say CONFIRM
#define REFRACTORY_MS      1000   // ignore all input after an actuation
#define MAX_REJECTS        3      // consecutive rejects before cooldown
#define COOLDOWN_MS        10000  // anti-babble mute period

// ---------------------------------------------------------------- SYSTEM
#define WATCHDOG_TIMEOUT_S 5
#define SERIAL_BAUD        921600
#define LOG_LEVEL          2      // 0=off 1=err 2=info 3=debug

// ---------------------------------------------------------------- CLASSES
// Order MUST match the Edge Impulse model output order exactly.
enum CommandId {
  CMD_START = 0,
  CMD_HALT,
  CMD_PAUSE,
  CMD_RESUME,
  CMD_FASTER,
  CMD_SLOWER,
  CMD_CONFIRM,
  CMD_CANCEL,
  CMD_UNKNOWN,
  CMD_SILENCE,
  CMD_COUNT
};

// Commands that start or increase motion require a CONFIRM handshake.
#define REQUIRES_CONFIRM(c) ((c) == CMD_START || (c) == CMD_RESUME)
// Commands blocked while the guard-door interlock is open.
#define BLOCKED_BY_INTERLOCK(c) ((c) == CMD_START || (c) == CMD_RESUME || (c) == CMD_FASTER)
