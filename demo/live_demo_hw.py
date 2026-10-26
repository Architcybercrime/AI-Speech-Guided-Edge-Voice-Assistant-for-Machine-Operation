"""
live_demo_hw.py - the live demo, now driving the real ESP32 hardware.

    python demo/live_demo_hw.py --port COM5
    python demo/live_demo_hw.py --port COM5 --no-voice      # keyboard-injected commands only

Keys (click the window first): E/G/R/Q as before, plus injected recognition for backup:
    1 wake  2 start  3 confirm  4 halt  5 pause  6 resume  7 faster  8 slower  9 cancel
Injected commands go through the SAME safety FSM and the SAME hardware. Only the
microphone->model step is skipped. Say so if you use it.
"""
import argparse
import json
import os
import queue
import sys
import tkinter as tk

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from live_demo import App, Listener, MODEL_PATH, LABELS_PATH   # noqa: E402
from intent_fsm import IntentFSM, Config, Tone                 # noqa: E402
from features import INPUT_SHAPE                               # noqa: E402
from hw_bridge import HardwareBridge, HardwareSync             # noqa: E402

TONE_CODE = {Tone.ACCEPT: 'A', Tone.REJECT: 'R', Tone.PROMPT: 'P',
             Tone.BOUNDARY: 'K', Tone.LOCKOUT: 'L'}
KEYMAP = {'1': 'hey_machina', '2': 'start', '3': 'confirm', '4': 'halt', '5': 'pause',
          '6': 'resume', '7': 'faster', '8': 'slower', '9': 'cancel'}


class NullListener:
    running = True
    error = None


class HwApp(App):
    def __init__(self, root, fsm, q, listener, bridge):
        super().__init__(root, fsm, q, listener)
        self.bridge = bridge
        self.sync = HardwareSync(fsm, bridge)

    def render_events(self, evs):
        super().render_events(evs)
        for ev in evs:
            code = TONE_CODE.get(ev.tone)
            if code:
                self.bridge.send('BEEP ' + code)
            if ev.message:
                self.sync.message = ev.message

    def render_panel(self):
        super().render_panel()
        evs = self.sync.step(self.now())
        if evs:
            self.render_events(evs)

    def on_key(self, e):
        k = (e.char or '').lower()
        if k in KEYMAP:
            for _ in range(self.fsm.cfg.vote_n):
                self.render_events(self.fsm.feed(KEYMAP[k], 0.99, self.now()))
            self.push('[keyboard] injected "%s"' % KEYMAP[k], 'state')
            return
        super().on_key(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', required=True, help='e.g. COM5')
    ap.add_argument('--threshold', type=float, default=0.80)
    ap.add_argument('--device', type=int, default=None)
    ap.add_argument('--no-voice', action='store_true')
    args = ap.parse_args()

    q = queue.Queue()
    if args.no_voice:
        listener = NullListener()
    else:
        labels = json.load(open(LABELS_PATH))
        print('loading model...')
        from tensorflow import keras
        model = keras.models.load_model(MODEL_PATH)
        model.predict(np.zeros((1,) + INPUT_SHAPE, np.float32), verbose=0)
        listener = Listener(model, labels, q, device=args.device)
        listener.start()

    print('opening', args.port, '...')
    bridge = HardwareBridge(args.port)
    fsm = IntentFSM(cfg=Config(conf_threshold=args.threshold))

    root = tk.Tk()
    HwApp(root, fsm, q, listener, bridge)
    root.mainloop()
    listener.running = False
    bridge.close()
    out = os.path.join(ROOT, 'models', 'demo_audit_log_hw.json')
    json.dump(fsm.log, open(out, 'w'), indent=2)
    print('audit log ->', out)


if __name__ == '__main__':
    main()
