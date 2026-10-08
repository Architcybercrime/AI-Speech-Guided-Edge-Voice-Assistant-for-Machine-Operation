"""
live_demo.py — the full system running on a laptop, no hardware required.

    microphone -> MFCC -> trained model -> safety FSM -> simulated machine

Run:
    python demo/live_demo.py

Keyboard (click the window first):
    E   toggle the hardwired EMERGENCY STOP
    G   toggle the guard-door interlock
    R   manual reset after lockout
    Q   quit

This is the same logic that will run on the ESP32-S3. The only difference is
that the relay, motor, LEDs and OLED are drawn on screen instead of wired.
"""

import json
import os
import queue
import sys
import threading
import time
import tkinter as tk

import numpy as np
import sounddevice as sd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from features import mfcc, SR, CLIP_LEN            # noqa: E402
from intent_fsm import IntentFSM, State, Tone      # noqa: E402

STRIDE_MS = 250
STRIDE = SR * STRIDE_MS // 1000
MODEL_PATH = os.path.join(ROOT, 'models', 'kws_model.keras')
LABELS_PATH = os.path.join(ROOT, 'models', 'labels.json')

BG, PANEL, FG, DIM = '#11181d', '#1b252c', '#eef3f6', '#7d8f9b'
AMBER, GREEN, RED, BLUE = '#e08900', '#2ecc71', '#e74c3c', '#3498db'


# ------------------------------------------------------------------ audio
class Listener(threading.Thread):
    """Captures audio and emits (label, confidence) at a fixed stride."""

    def __init__(self, model, labels, out_q):
        super().__init__(daemon=True)
        self.model, self.labels, self.out_q = model, labels, out_q
        self.ring = np.zeros(CLIP_LEN, np.float32)
        self.audio_q = queue.Queue()
        self.level = 0.0
        self.running = True

    def _cb(self, indata, frames, t, status):
        self.audio_q.put(indata[:, 0].copy())

    def run(self):
        with sd.InputStream(samplerate=SR, channels=1, dtype='float32',
                            blocksize=STRIDE, callback=self._cb):
            while self.running:
                try:
                    block = self.audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                n = len(block)
                self.ring = np.concatenate([self.ring[n:], block])
                self.level = float(np.sqrt(np.mean(block ** 2)))
                t0 = time.perf_counter()
                feat = mfcc(self.ring)[None, ..., None]
                probs = self.model.predict(feat, verbose=0)[0]
                ms = (time.perf_counter() - t0) * 1000
                i = int(probs.argmax())
                self.out_q.put((self.labels[i], float(probs[i]), ms, self.level))


# ------------------------------------------------------------------ ui
class App:
    def __init__(self, root, fsm, q, listener):
        self.fsm, self.q, self.listener = fsm, q, listener
        self.t0 = time.time()
        self.latencies = []
        self.events = []

        root.title('Edge Voice Assistant - live demo')
        root.configure(bg=BG)
        root.geometry('980x620')
        self.root = root

        tk.Label(root, text='AI SPEECH-GUIDED EDGE VOICE ASSISTANT', bg=BG, fg=AMBER,
                 font=('Consolas', 11, 'bold')).pack(pady=(14, 0))
        tk.Label(root, text='offline  ·  on-device  ·  no network in the signal path',
                 bg=BG, fg=DIM, font=('Consolas', 9)).pack()

        body = tk.Frame(root, bg=BG); body.pack(fill='both', expand=True, padx=16, pady=12)
        left = tk.Frame(body, bg=BG); left.pack(side='left', fill='both', expand=True)
        right = tk.Frame(body, bg=BG); right.pack(side='right', fill='both', padx=(14, 0))

        # ---- state banner
        self.state_lbl = tk.Label(left, text='IDLE', bg=PANEL, fg=FG,
                                  font=('Consolas', 26, 'bold'), pady=14)
        self.state_lbl.pack(fill='x')

        # ---- OLED mock
        oled = tk.Frame(left, bg='#04131a', highlightbackground='#2b3942',
                        highlightthickness=1)
        oled.pack(fill='x', pady=10)
        tk.Label(oled, text='OLED', bg='#04131a', fg=DIM,
                 font=('Consolas', 8)).pack(anchor='w', padx=8, pady=(5, 0))
        self.oled1 = tk.Label(oled, text='Say "Hey Machina"', bg='#04131a', fg='#5df2c8',
                              font=('Consolas', 15, 'bold'), anchor='w')
        self.oled1.pack(fill='x', padx=8)
        self.oled2 = tk.Label(oled, text='', bg='#04131a', fg='#5df2c8',
                              font=('Consolas', 11), anchor='w')
        self.oled2.pack(fill='x', padx=8, pady=(0, 8))

        # ---- heard
        hf = tk.Frame(left, bg=PANEL); hf.pack(fill='x', pady=(0, 10))
        tk.Label(hf, text='HEARD', bg=PANEL, fg=DIM,
                 font=('Consolas', 8)).pack(anchor='w', padx=10, pady=(6, 0))
        self.heard = tk.Label(hf, text='—', bg=PANEL, fg=FG,
                              font=('Consolas', 17, 'bold'), anchor='w')
        self.heard.pack(fill='x', padx=10)
        self.conf_c = tk.Canvas(hf, height=12, bg='#243039', highlightthickness=0)
        self.conf_c.pack(fill='x', padx=10, pady=(4, 4))
        self.conf_txt = tk.Label(hf, text='confidence 0.00   ·   threshold 0.85',
                                 bg=PANEL, fg=DIM, font=('Consolas', 9), anchor='w')
        self.conf_txt.pack(fill='x', padx=10, pady=(0, 8))

        # ---- machine outputs
        mf = tk.Frame(left, bg=PANEL); mf.pack(fill='x')
        tk.Label(mf, text='MACHINE OUTPUTS', bg=PANEL, fg=DIM,
                 font=('Consolas', 8)).pack(anchor='w', padx=10, pady=(6, 2))
        row = tk.Frame(mf, bg=PANEL); row.pack(fill='x', padx=10, pady=(0, 8))
        self.motor_c = tk.Canvas(row, width=92, height=92, bg=PANEL, highlightthickness=0)
        self.motor_c.pack(side='left')
        info = tk.Frame(row, bg=PANEL); info.pack(side='left', padx=14)
        self.relay_lbl = tk.Label(info, text='RELAY 1   OPEN', bg=PANEL, fg=DIM,
                                  font=('Consolas', 12, 'bold'), anchor='w')
        self.relay_lbl.pack(anchor='w')
        self.speed_lbl = tk.Label(info, text='SPEED  3/5    PWM duty 0', bg=PANEL, fg=FG,
                                  font=('Consolas', 12), anchor='w')
        self.speed_lbl.pack(anchor='w', pady=3)
        self.led_c = tk.Canvas(info, width=200, height=30, bg=PANEL, highlightthickness=0)
        self.led_c.pack(anchor='w')
        self.angle = 0.0

        # ---- right column: safety + log
        sf = tk.Frame(right, bg=PANEL, width=330); sf.pack(fill='x')
        tk.Label(sf, text='SAFETY INPUTS  (hardwired)', bg=PANEL, fg=DIM,
                 font=('Consolas', 8)).pack(anchor='w', padx=10, pady=(6, 2))
        self.estop_lbl = tk.Label(sf, text='[E]  E-STOP      OK', bg=PANEL, fg=GREEN,
                                  font=('Consolas', 11, 'bold'), anchor='w')
        self.estop_lbl.pack(fill='x', padx=10)
        self.guard_lbl = tk.Label(sf, text='[G]  GUARD DOOR  CLOSED', bg=PANEL, fg=GREEN,
                                  font=('Consolas', 11, 'bold'), anchor='w')
        self.guard_lbl.pack(fill='x', padx=10, pady=(2, 2))
        tk.Label(sf, text='[R]  reset      [Q]  quit', bg=PANEL, fg=DIM,
                 font=('Consolas', 9), anchor='w').pack(fill='x', padx=10, pady=(0, 8))

        lf = tk.Frame(right, bg=PANEL); lf.pack(fill='both', expand=True, pady=(10, 0))
        tk.Label(lf, text='AUDIT LOG', bg=PANEL, fg=DIM,
                 font=('Consolas', 8)).pack(anchor='w', padx=10, pady=(6, 2))
        self.log = tk.Text(lf, bg='#121b21', fg=FG, font=('Consolas', 9), width=42,
                           height=18, bd=0, highlightthickness=0, wrap='word')
        self.log.pack(fill='both', expand=True, padx=10, pady=(0, 8))
        for tag, col in (('accepted', GREEN), ('rejected', RED),
                         ('prompt', AMBER), ('safety', RED), ('state', DIM)):
            self.log.tag_config(tag, foreground=col)

        self.stats = tk.Label(root, text='', bg=BG, fg=DIM, font=('Consolas', 9))
        self.stats.pack(pady=(0, 8))

        root.bind('<Key>', self.on_key)
        root.after(40, self.loop)

    # -------------------------------------------------------------- input
    def now(self):
        return int((time.time() - self.t0) * 1000)

    def on_key(self, e):
        k = e.char.lower()
        t = self.now()
        if k == 'e':
            evs = self.fsm.set_estop(not self.fsm.estop_tripped, t)
            if not self.fsm.estop_tripped:
                self.push([], 'E-stop released - press R to reset', 'state')
            self.render_events(evs)
        elif k == 'g':
            self.render_events(self.fsm.set_interlock(not self.fsm.interlock_open, t))
            if not self.fsm.interlock_open:
                self.push([], 'Guard door closed', 'state')
        elif k == 'r':
            self.render_events(self.fsm.reset(t))
        elif k == 'q':
            self.listener.running = False
            self.root.destroy()

    def push(self, _unused, text, tag):
        ts = time.strftime('%H:%M:%S')
        self.log.insert('end', f'{ts}  {text}\n', tag)
        self.log.see('end')

    def render_events(self, evs):
        for ev in evs:
            tag = ev.kind if ev.kind in ('accepted', 'rejected', 'prompt', 'safety') else 'state'
            txt = ev.message or ev.label
            if ev.label and ev.kind in ('accepted', 'rejected', 'prompt'):
                txt = f'{ev.label.upper():<12} {txt}'
            self.push(None, txt, tag)
            if ev.kind in ('accepted', 'prompt'):
                self.oled1.config(text=(ev.label or '').upper() or 'OK')
                self.oled2.config(text=ev.message[:34])
            elif ev.kind == 'rejected':
                self.oled1.config(text='REJECTED')
                self.oled2.config(text=ev.message[:34])
            elif ev.kind == 'safety':
                self.oled1.config(text=ev.label or 'SAFETY')
                self.oled2.config(text=ev.message[:34])
            self.events.append(ev)

    # -------------------------------------------------------------- loop
    def loop(self):
        t = self.now()
        self.render_events(self.fsm.tick(t))

        drained = 0
        while not self.q.empty() and drained < 6:
            label, conf, ms, level = self.q.get()
            self.latencies.append(ms)
            drained += 1
            self.heard.config(text=label.upper() if conf >= 0.5 else '—')
            w = max(1, int(self.conf_c.winfo_width() * min(conf, 1.0)))
            self.conf_c.delete('all')
            colour = GREEN if conf >= self.fsm.cfg.conf_threshold else DIM
            self.conf_c.create_rectangle(0, 0, w, 12, fill=colour, width=0)
            thr_x = int(self.conf_c.winfo_width() * self.fsm.cfg.conf_threshold)
            self.conf_c.create_line(thr_x, 0, thr_x, 12, fill=AMBER, width=2)
            self.conf_txt.config(text=f'confidence {conf:.2f}   ·   threshold '
                                      f'{self.fsm.cfg.conf_threshold:.2f}   ·   '
                                      f'inference {ms:.0f} ms')
            self.render_events(self.fsm.feed(label, conf, self.now()))

        self.render_panel()
        self.root.after(40, self.loop)

    def render_panel(self):
        s = self.fsm.status()
        colours = {'IDLE': DIM, 'LISTENING': BLUE, 'AWAIT_CONFIRM': AMBER,
                   'COOLDOWN': AMBER, 'LOCKOUT': RED}
        self.state_lbl.config(text=s['state'], fg=colours.get(s['state'], FG))

        if s['state'] == 'IDLE' and not s['motor_on']:
            self.oled1.config(text='Say "Hey Machina"')

        self.relay_lbl.config(text=f"RELAY 1   {'CLOSED' if s['relay_enable'] else 'OPEN'}",
                              fg=GREEN if s['relay_enable'] else DIM)
        self.speed_lbl.config(text=f"SPEED  {s['speed']}/5    PWM duty {s['duty']}")

        # motor
        c = self.motor_c
        c.delete('all')
        c.create_oval(6, 6, 86, 86, outline='#39474f', width=2)
        if s['motor_on']:
            self.angle += 0.08 * s['speed']
        import math
        for k in range(3):
            a = self.angle + k * 2 * math.pi / 3
            c.create_line(46, 46, 46 + 34 * math.cos(a), 46 + 34 * math.sin(a),
                          fill=GREEN if s['motor_on'] else '#39474f', width=4)
        c.create_oval(40, 40, 52, 52, fill=GREEN if s['motor_on'] else '#39474f', width=0)

        # LEDs
        lc = self.led_c
        lc.delete('all')
        st = s['state']
        leds = [('G', GREEN, st == 'IDLE' and not s['estop']),
                ('B', BLUE, st in ('LISTENING', 'AWAIT_CONFIRM')),
                ('R', RED, st == 'LOCKOUT' or s['estop'])]
        for i, (_n, col, on) in enumerate(leds):
            x = 8 + i * 34
            lc.create_oval(x, 6, x + 20, 26, fill=col if on else '#243039', width=0)

        self.estop_lbl.config(text=f"[E]  E-STOP      {'TRIPPED' if s['estop'] else 'OK'}",
                              fg=RED if s['estop'] else GREEN)
        self.guard_lbl.config(
            text=f"[G]  GUARD DOOR  {'OPEN' if s['interlock_open'] else 'CLOSED'}",
            fg=RED if s['interlock_open'] else GREEN)

        if self.latencies:
            arr = np.array(self.latencies[-200:])
            acc = sum(1 for e in self.events if e.kind == 'accepted')
            rej = sum(1 for e in self.events if e.kind == 'rejected')
            self.stats.config(text=f'inference p95 {np.percentile(arr, 95):.0f} ms   ·   '
                                   f'windows {len(self.latencies)}   ·   '
                                   f'accepted {acc}   ·   rejected {rej}   ·   '
                                   f'uptime {int(time.time()-self.t0)} s')


def main():
    if not os.path.exists(MODEL_PATH):
        sys.exit(f'Model not found at {MODEL_PATH}\nRun: python tools/train_model.py')
    labels = json.load(open(LABELS_PATH))
    print('loading model...')
    from tensorflow import keras
    model = keras.models.load_model(MODEL_PATH)
    model.predict(np.zeros((1,) + model.input_shape[1:], np.float32), verbose=0)  # warm up

    q = queue.Queue()
    fsm = IntentFSM()
    listener = Listener(model, labels, q)
    listener.start()

    root = tk.Tk()
    App(root, fsm, q, listener)
    root.mainloop()
    listener.running = False

    # dump the audit trail for the report
    os.makedirs(os.path.join(ROOT, 'models'), exist_ok=True)
    out = os.path.join(ROOT, 'models', 'demo_audit_log.json')
    json.dump(fsm.log, open(out, 'w'), indent=2)
    print(f'audit log -> {out}')


if __name__ == '__main__':
    main()
