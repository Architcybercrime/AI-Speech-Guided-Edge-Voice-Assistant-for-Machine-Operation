"""
hw_bridge.py - laptop <-> ESP32 link for the voice assistant.

The laptop recognises speech and runs the safety FSM. The ESP32 owns the physical
outputs and has the final say: its E-stop, guard switch and host-watchdog act locally,
without waiting for the laptop. Hardware events are fed back into the FSM.
"""
import queue
import threading
import time

import serial


class HardwareBridge:
    def __init__(self, port, baud=115200, boot_wait=2.0):
        # opening the port toggles DTR/RTS and resets a DevKit, so give it time to boot
        self.ser = serial.serial_for_url(port, baudrate=baud, timeout=0.1)
        self.events = queue.Queue()
        self.state = {'es': 0, 'il': 1, 'run': 0, 'latched': 0}
        self.seen_status = False
        self._lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()
        time.sleep(boot_wait)
        self.send('WD 1')                    # arm the ESP32's host watchdog

    def send(self, line):
        try:
            with self._lock:
                self.ser.write((line + '\n').encode('ascii', 'replace'))
        except Exception:
            pass

    def _reader(self):
        buf = b''
        while self.running:
            try:
                data = self.ser.read(128)
            except Exception:
                break
            if not data:
                continue
            buf += data
            while b'\n' in buf:
                raw, buf = buf.split(b'\n', 1)
                self._parse(raw.decode('ascii', 'ignore').strip())

    def _parse(self, line):
        p = line.split()
        if not p:
            return
        if p[0] == 'S' and len(p) >= 5:
            try:
                es, il, run, latched = (int(x) for x in p[1:5])
            except ValueError:
                return
            old = dict(self.state)
            self.state = {'es': es, 'il': il, 'run': run, 'latched': latched}
            if not self.seen_status or es != old['es']:
                self.events.put('ES%d' % es)
            if not self.seen_status or il != old['il']:
                self.events.put('IL%d' % il)
            self.seen_status = True
        elif p[0] == 'RST':
            self.events.put('RST')
        elif p[0] == 'WDOG':
            self.events.put('WDOG')

    def poll(self):
        out = []
        while not self.events.empty():
            out.append(self.events.get())
        return out

    def close(self):
        self.running = False
        try:
            self.send('STOP')
            self.ser.close()
        except Exception:
            pass


class HardwareSync:
    """Pushes FSM state out to the hardware and pulls hardware events into the FSM."""

    LED = {'IDLE': 'G', 'LISTENING': 'B', 'AWAIT_CONFIRM': 'B', 'COOLDOWN': 'Y', 'LOCKOUT': 'R'}

    def __init__(self, fsm, bridge):
        self.fsm, self.bridge = fsm, bridge
        self.sent = {}
        self.message = 'Say Hey Machina'
        self.t_ka = 0
        self.log = []

    def _send_if_changed(self, key, value, line):
        if self.sent.get(key) != value:
            self.sent[key] = value
            self.bridge.send(line)

    def step(self, now_ms):
        events = []
        for ev in self.bridge.poll():
            if ev == 'ES1':
                events += self.fsm.set_estop(True, now_ms)
            elif ev == 'ES0':
                events += self.fsm.set_estop(False, now_ms)
            elif ev == 'IL1':
                events += self.fsm.set_interlock(True, now_ms)
            elif ev == 'IL0':
                events += self.fsm.set_interlock(False, now_ms)
            elif ev == 'RST':
                events += self.fsm.reset(now_ms)

        s = self.fsm.status()
        run = s['speed'] if (s['motor_on'] and not s['estop'] and not s['interlock_open']) else 0
        self._send_if_changed('run', run, 'RUN %d' % run if run else 'STOP')
        self._send_if_changed('led', self.LED.get(s['state'], 'X'), 'LED %s' % self.LED.get(s['state'], 'X'))
        line1 = s['state'] if not run else 'RUN %d/5' % run
        text = ('%s|%s' % (line1, self.message)).encode('ascii', 'replace').decode()[:60]
        self._send_if_changed('oled', text, 'OLED ' + text)

        if now_ms - self.t_ka >= 400:        # keep the ESP32 watchdog fed
            self.t_ka = now_ms
            self.bridge.send('KA')
        return events
