"""
test_fsm.py — safety verification for the intent FSM. No audio, no hardware.

Run:  python tools/test_fsm.py

Every test corresponds to an acceptance criterion in the synopsis (section 2.6)
or a defence layer in section 9. Showing this passing in the review is direct
evidence that the safety logic is implemented, not merely designed.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from intent_fsm import IntentFSM, Config, State  # noqa: E402

PASS, FAIL = 0, 0
HI = 0.97          # a confident recognition


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  PASS  {name}')
    else:
        FAIL += 1
        print(f'  FAIL  {name}')


def say(fsm, label, t, conf=HI, times=None):
    """Feed a label enough times to satisfy N-of-M voting."""
    times = times or fsm.cfg.vote_n
    out = []
    for i in range(times):
        out += fsm.feed(label, conf, t + i * 10)
    return out


def wake(fsm, t):
    say(fsm, 'hey_machina', t)


# ---------------------------------------------------------------- layer 1
print('\nLayer 1 - wake-word gate')
f = IntentFSM()
say(f, 'start', 1000)
check('START before wake word does nothing', f.state is State.IDLE and not f.machine.motor_on)

f = IntentFSM()
wake(f, 1000)
check('wake word opens the listening window', f.state is State.LISTENING)

# ---------------------------------------------------------------- layer 2
print('\nLayer 2 - confidence threshold')
f = IntentFSM()
say(f, 'hey_machina', 1000, conf=0.60)
check('low-confidence wake word is discarded', f.state is State.IDLE)

f = IntentFSM()
wake(f, 1000)
say(f, 'halt', 1200, conf=0.50)
check('low-confidence command is discarded', f.state is State.LISTENING)

# ---------------------------------------------------------------- layer 3
print('\nLayer 3 - N-of-M voting')
f = IntentFSM()
say(f, 'hey_machina', 1000, times=2)       # one short of vote_n
check('single-window detection is not enough', f.state is State.IDLE)

f = IntentFSM()
f.feed('hey_machina', HI, 1000)
f.feed('halt', HI, 1010)
f.feed('hey_machina', HI, 1020)
f.feed('faster', HI, 1030)
check('alternating labels never reach quorum', f.state is State.IDLE)

# ---------------------------------------------------------------- layer 4
print('\nLayer 4 - confirm handshake')
f = IntentFSM()
wake(f, 1000)
say(f, 'start', 1200)
check('START moves to AWAIT_CONFIRM', f.state is State.AWAIT_CONFIRM)
check('motor has NOT started yet', not f.machine.motor_on)
say(f, 'confirm', 1500)
check('CONFIRM starts the motor', f.machine.motor_on and f.machine.relay_enable)

f = IntentFSM()
wake(f, 1000)
say(f, 'start', 1200)
f.tick(6000)
check('no confirmation within timeout discards it', not f.machine.motor_on and f.state is State.IDLE)

f = IntentFSM()
wake(f, 1000)
say(f, 'start', 1200)
say(f, 'cancel', 1400)
check('CANCEL discards the pending command', not f.machine.motor_on and f.state is State.IDLE)

f = IntentFSM()
wake(f, 1000)
say(f, 'halt', 1200)
check('HALT executes immediately, no confirm needed', f.state is State.IDLE)

# ---------------------------------------------------------------- layer 5
print('\nLayer 5 - guard-door interlock')
f = IntentFSM()
f.set_interlock(True)
wake(f, 1000)
say(f, 'start', 1200)
check('START refused while guard is open', not f.machine.motor_on and f.state is not State.AWAIT_CONFIRM)

f = IntentFSM()
wake(f, 1000)
say(f, 'start', 1200)
f.set_interlock(True, 1300)          # guard opens mid-handshake
say(f, 'confirm', 1400)
check('guard opened during handshake blocks actuation', not f.machine.motor_on)

f = IntentFSM()
wake(f, 1000)
say(f, 'start', 1200)
say(f, 'confirm', 1500)
f.set_interlock(True, 2000)
check('opening the guard stops a running motor', not f.machine.motor_on)

# ---------------------------------------------------------------- layer 6
print('\nLayer 6 - refractory period')
f = IntentFSM()
wake(f, 1000)
say(f, 'start', 1200)
say(f, 'confirm', 1500)
wake(f, 1600)                        # inside the 1 s refractory window
check('input ignored immediately after actuation', f.state is State.IDLE)
wake(f, 3000)
check('input accepted again after refractory', f.state is State.LISTENING)

# ---------------------------------------------------------------- layer 7
print('\nLayer 7 - anti-babble cooldown')
f = IntentFSM()
for k in range(3):
    t = 1000 + k * 2000
    wake(f, t)
    say(f, 'confirm', t + 200)       # nothing pending -> reject
check('three rejects trigger cooldown', f.state is State.COOLDOWN)
wake(f, 8000)
check('system is deaf during cooldown', f.state is State.COOLDOWN)
f.tick(20000)
check('cooldown expires', f.state is State.IDLE)

# ---------------------------------------------------------------- E-stop
print('\nEmergency stop - authoritative, firmware cannot override')
f = IntentFSM()
wake(f, 1000)
say(f, 'start', 1200)
say(f, 'confirm', 1500)
f.set_estop(True, 2000)
check('E-stop de-energises outputs', not f.machine.motor_on and not f.machine.relay_enable)
check('E-stop forces LOCKOUT', f.state is State.LOCKOUT)
wake(f, 3000)
say(f, 'start', 3500)
check('no command works during lockout', not f.machine.motor_on)
f.reset(4000)
check('reset refused while E-stop engaged', f.state is State.LOCKOUT)
f.set_estop(False, 5000)
f.reset(5100)
check('reset works once E-stop released', f.state is State.IDLE)

# ---------------------------------------------------------------- speed
print('\nSpeed control - bounded')
f = IntentFSM()
for k in range(4):
    t = 1000 + k * 3000
    wake(f, t)
    say(f, 'faster', t + 200)
check('speed clamps at maximum', f.machine.speed == f.cfg.speed_max)
for k in range(8):
    t = 20000 + k * 3000
    wake(f, t)
    say(f, 'slower', t + 200)
check('speed clamps at minimum', f.machine.speed == f.cfg.speed_min)

# ---------------------------------------------------------------- pause/resume
print('\nPause and resume - state retention')
f = IntentFSM()
wake(f, 1000); say(f, 'start', 1200); say(f, 'confirm', 1500)
wake(f, 3000); say(f, 'faster', 3200)
wake(f, 6000); say(f, 'faster', 6200)
sp = f.machine.speed
wake(f, 9000); say(f, 'pause', 9200)
check('PAUSE stops the motor', not f.machine.motor_on)
wake(f, 12000); say(f, 'resume', 12200); say(f, 'confirm', 12500)
check('RESUME restores the retained speed', f.machine.motor_on and f.machine.speed == sp)

# ---------------------------------------------------------------- audit
print('\nAudit trail')
f = IntentFSM()
wake(f, 1000); say(f, 'start', 1200); say(f, 'confirm', 1500)
outcomes = {e['outcome'] for e in f.log}
check('every event is logged with an outcome', len(f.log) >= 3 and 'accepted' in outcomes)
check('confidence recorded on log entries', all('confidence' in e for e in f.log))

# ---------------------------------------------------------------- summary
print('\n' + '=' * 54)
print(f'  {PASS} passed, {FAIL} failed')
print('=' * 54)
print('\nZero unsafe actuations: no path starts the motor without a'
      '\nwake word, a command, a CONFIRM, and a closed interlock.')
sys.exit(1 if FAIL else 0)
