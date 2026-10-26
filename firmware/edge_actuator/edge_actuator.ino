/*
 * edge_actuator.ino  -  ESP32 DevKit V1 (classic ESP32), Arduino core 2.x or 3.x
 *
 * The ESP32 owns everything physical and everything safety-related:
 *   - E-stop button  -> outputs drop IMMEDIATELY and LATCH (needs RESET to clear)
 *   - Guard switch   -> motion refused / stopped while the guard is open
 *   - Host watchdog  -> motor stops if the laptop goes silent for 1.5 s
 * The laptop only *requests* motion. The ESP32 may refuse.
 *
 * Serial protocol, 115200 baud, newline terminated.
 *   host -> ESP32:  PING | KA | WD 0/1 | RUN 0-5 | STOP | BEEP A/R/P/L/K
 *                   LED R/G/B/Y/X | OLED line1|line2
 *   ESP32 -> host:  READY | PONG | RST | WDOG | ERR
 *                   S <estop> <guard_open> <speed> <latched>   (every 200 ms)
 *
 * Test by hand in the Arduino Serial Monitor (set "Newline"): type  RUN 3
 * The host watchdog is OFF until the host sends "WD 1", so manual testing works.
 */
#include <Arduino.h>
#include <Wire.h>
#include <string.h>
#include <stdlib.h>
#include <strings.h>

// ------------------------------------------------------------ switches
#define USE_OLED          1   // needs libs: "Adafruit SSD1306" + "Adafruit GFX". Set 0 to skip.
#define USE_L298N         1   // 1: PWM speed through L298N.  0: relay switches the motor directly.
#define RELAY_OFF_HIZ     0   // set 1 if the relay stays ON when its pin is driven HIGH
#define LED_COMMON_ANODE  0   // set 1 if the RGB LED's long leg goes to 3V3 instead of GND

#if USE_OLED
  #include <Adafruit_GFX.h>
  #include <Adafruit_SSD1306.h>
  Adafruit_SSD1306 oled(128, 64, &Wire, -1);
  static bool oledOk = false;
#endif

// ------------------------------------------------------------ pins
// Deliberately avoids strapping pins 0/2/5/12/15 so the board always boots.
const uint8_t PIN_SDA = 21, PIN_SCL = 22;                    // OLED
const uint8_t PIN_RELAY = 32;                                // relay IN1, active LOW
const uint8_t PIN_IN1 = 16, PIN_IN2 = 17, PIN_ENA = 27;      // L298N
const uint8_t PIN_BUZZ = 23;
const uint8_t PIN_LED_R = 18, PIN_LED_G = 19, PIN_LED_B = 33;
const uint8_t PIN_ESTOP = 4;      // tactile button to GND, pressed = tripped
const uint8_t PIN_RESET = 14;     // tactile button to GND
const uint8_t PIN_GUARD = 13;     // jumper wire to GND = guard closed, wire pulled = OPEN

const uint8_t DUTY[6] = {0, 90, 130, 165, 205, 255};

// ------------------------------------------------------------ state
static bool     latched = false;
static uint8_t  runSpeed = 0;
static bool     wdOn = false;
static uint32_t lastHostMs = 0;
static uint32_t lastStatusMs = 0;

// ------------------------------------------------------------ forward decls
void relayWrite(bool on);
void motorOff();
void motorRun(int speed);
void setLed(char c);
void startBeep(const uint16_t* p);
void showOled(const char* a, const char* b);
void handleLine(char* line);

#if ESP_ARDUINO_VERSION_MAJOR >= 3
  #define PWM_ATTACH() ledcAttach(PIN_ENA, 2000, 8)
  #define PWM_WRITE(v) ledcWrite(PIN_ENA, (v))
#else
  #define PWM_ATTACH() do { ledcSetup(0, 2000, 8); ledcAttachPin(PIN_ENA, 0); } while (0)
  #define PWM_WRITE(v) ledcWrite(0, (v))
#endif

// ------------------------------------------------------------ debounced inputs
struct Debounced {
  uint8_t pin; bool active; bool raw; uint32_t t;   // active == pin reads LOW
  void begin(uint8_t p) {
    pin = p; pinMode(pin, INPUT_PULLUP);
    raw = active = (digitalRead(pin) == LOW); t = millis();
  }
  bool update() {                                   // true when 'active' changed
    bool r = (digitalRead(pin) == LOW);
    if (r != raw) { raw = r; t = millis(); }
    if (raw != active && millis() - t >= 25) { active = raw; return true; }
    return false;
  }
};
static Debounced btnEstop, btnReset, swGuard;       // swGuard.active == guard CLOSED

// ------------------------------------------------------------ buzzer (non-blocking)
// durations in ms, alternating ON/OFF, 0 terminates
static const uint16_t PAT_A[] = {70, 50, 70, 0};            // accepted
static const uint16_t PAT_R[] = {350, 0};                   // rejected
static const uint16_t PAT_P[] = {60, 60, 60, 60, 60, 0};    // prompt / listening
static const uint16_t PAT_L[] = {500, 100, 500, 0};         // lockout
static const uint16_t PAT_K[] = {40, 0};                    // boundary tick
static const uint16_t* pat = NULL;
static uint8_t  patIdx = 0;
static uint32_t patT = 0;

void startBeep(const uint16_t* p) {
  pat = p; patIdx = 0; patT = millis(); digitalWrite(PIN_BUZZ, HIGH);
}
void updateBuzzer() {
  if (!pat) return;
  if (millis() - patT >= pat[patIdx]) {
    patIdx++; patT = millis();
    if (pat[patIdx] == 0) { digitalWrite(PIN_BUZZ, LOW); pat = NULL; }
    else digitalWrite(PIN_BUZZ, (patIdx % 2 == 0) ? HIGH : LOW);
  }
}

// ------------------------------------------------------------ LED
void setLed(char c) {
  bool r = false, g = false, b = false;
  switch (c) {
    case 'R': r = true; break;
    case 'G': g = true; break;
    case 'B': b = true; break;
    case 'Y': r = true; g = true; break;
    default: break;                                  // 'X' = off
  }
#if LED_COMMON_ANODE
  r = !r; g = !g; b = !b;
#endif
  digitalWrite(PIN_LED_R, r ? HIGH : LOW);
  digitalWrite(PIN_LED_G, g ? HIGH : LOW);
  digitalWrite(PIN_LED_B, b ? HIGH : LOW);
}

// ------------------------------------------------------------ OLED
void showOled(const char* a, const char* b) {
#if USE_OLED
  if (!oledOk) return;
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(2);
  oled.setCursor(0, 0);
  oled.println(a);
  oled.setTextSize(1);
  oled.setCursor(0, 30);
  oled.println(b);
  oled.display();
#else
  (void)a; (void)b;
#endif
}

// ------------------------------------------------------------ relay + motor
void relayWrite(bool on) {
  if (on) {
    pinMode(PIN_RELAY, OUTPUT);
    digitalWrite(PIN_RELAY, LOW);                    // active LOW
  } else {
#if RELAY_OFF_HIZ
    pinMode(PIN_RELAY, INPUT);
#else
    pinMode(PIN_RELAY, OUTPUT);
    digitalWrite(PIN_RELAY, HIGH);
#endif
  }
}

void motorOff() {
  relayWrite(false);                                 // cut power first
#if USE_L298N
  PWM_WRITE(0);
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, LOW);
#endif
  runSpeed = 0;
}

void motorRun(int speed) {
  if (speed > 5) speed = 5;
  bool guardClosed = swGuard.active;
  if (speed <= 0 || latched || btnEstop.active || !guardClosed) { motorOff(); return; }
#if USE_L298N
  digitalWrite(PIN_IN1, HIGH);
  digitalWrite(PIN_IN2, LOW);
  PWM_WRITE(DUTY[speed]);
#endif
  relayWrite(true);                                  // power last
  runSpeed = (uint8_t)speed;
}

// ------------------------------------------------------------ serial
static char    lineBuf[96];
static uint8_t lineLen = 0;

void handleLine(char* line) {
  lastHostMs = millis();
  char* cmd = strtok(line, " ");
  if (!cmd) return;
  char* arg = strtok(NULL, "");
  for (char* p = cmd; *p; p++) if (*p >= 'a' && *p <= 'z') *p -= 32;

  if      (!strcmp(cmd, "PING")) Serial.println("PONG");
  else if (!strcmp(cmd, "KA"))   { /* liveness only */ }
  else if (!strcmp(cmd, "WD"))   wdOn = (arg && atoi(arg) != 0);
  else if (!strcmp(cmd, "RUN"))  motorRun(arg ? atoi(arg) : 0);
  else if (!strcmp(cmd, "STOP")) motorOff();
  else if (!strcmp(cmd, "BEEP") && arg) {
    switch (arg[0]) {
      case 'A': case 'a': startBeep(PAT_A); break;
      case 'R': case 'r': startBeep(PAT_R); break;
      case 'P': case 'p': startBeep(PAT_P); break;
      case 'L': case 'l': startBeep(PAT_L); break;
      case 'K': case 'k': startBeep(PAT_K); break;
      default: break;
    }
  }
  else if (!strcmp(cmd, "LED") && arg) {
    char c = arg[0]; if (c >= 'a' && c <= 'z') c -= 32;
    setLed(c);
  }
  else if (!strcmp(cmd, "OLED") && arg) {
    char* bar = strchr(arg, '|');
    if (bar) { *bar = 0; showOled(arg, bar + 1); }
    else     showOled(arg, "");
  }
  else Serial.println("ERR");
}

void readSerial() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      lineBuf[lineLen] = 0;
      if (lineLen) handleLine(lineBuf);
      lineLen = 0;
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = c;
    }
  }
}

// ------------------------------------------------------------ setup / loop
void setup() {
  relayWrite(false);                                 // first thing: everything OFF
  pinMode(PIN_IN1, OUTPUT); pinMode(PIN_IN2, OUTPUT);
  digitalWrite(PIN_IN1, LOW); digitalWrite(PIN_IN2, LOW);
  PWM_ATTACH(); PWM_WRITE(0);
  pinMode(PIN_BUZZ, OUTPUT); digitalWrite(PIN_BUZZ, LOW);
  pinMode(PIN_LED_R, OUTPUT); pinMode(PIN_LED_G, OUTPUT); pinMode(PIN_LED_B, OUTPUT);
  setLed('X');

  Serial.begin(115200);
  btnEstop.begin(PIN_ESTOP);
  btnReset.begin(PIN_RESET);
  swGuard.begin(PIN_GUARD);
  if (btnEstop.active) latched = true;               // booted with E-stop held: stay locked

  Wire.begin(PIN_SDA, PIN_SCL);
#if USE_OLED
  oledOk = oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
#endif
  showOled("READY", "waiting for host");
  setLed(latched ? 'R' : 'G');
  lastHostMs = millis();
  Serial.println("READY");
}

void loop() {
  readSerial();
  updateBuzzer();

  // ---- E-stop: highest priority, handled before anything else
  if (btnEstop.update() && btnEstop.active) {
    latched = true;
    motorOff();
    setLed('R');
    startBeep(PAT_L);
    showOled("E-STOP", "press RESET after release");
  }

  // ---- Reset: only clears the latch once the E-stop is released
  if (btnReset.update() && btnReset.active) {
    if (btnEstop.active)  { startBeep(PAT_R); }
    else if (latched)     { latched = false; setLed('G'); startBeep(PAT_A);
                            showOled("RESET", "armed"); Serial.println("RST"); }
  }

  // ---- Guard switch
  if (swGuard.update() && !swGuard.active) {          // guard just opened
    if (runSpeed > 0) { motorOff(); startBeep(PAT_R); showOled("GUARD OPEN", "motion stopped"); }
  }

  // ---- Host watchdog
  if (wdOn && runSpeed > 0 && millis() - lastHostMs > 1500) {
    motorOff(); setLed('R'); startBeep(PAT_L);
    Serial.println("WDOG");
  }

  // ---- Status report
  if (millis() - lastStatusMs >= 200) {
    lastStatusMs = millis();
    Serial.printf("S %d %d %d %d\n", btnEstop.active ? 1 : 0, swGuard.active ? 0 : 1,
                  (int)runSpeed, latched ? 1 : 0);
  }
}
