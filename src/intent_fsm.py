"""
intent_fsm.py — safety state machine for the edge voice assistant.

Pure logic: no audio, no hardware, no model. Feed it (label, confidence)
and it decides whether anything is allowed to move. This mirrors exactly
what will run in C on the ESP32-S3, so it can be unit-tested on a laptop
before any component exists.

Seven defence layers between a sound and a moving machine:
  1. wake-word gate        - commands only heard after explicit activation
  2. confidence threshold  - softmax below CONF_THRESHOLD is discarded
  3. N-of-M voting         - N of the last M windows must agree
  4. confirm handshake     - motion needs a second, separate utterance
  5. interlock check       - guard door must read closed
  6. refractory period     - input ignored briefly after any actuation
  7. anti-babble cooldown  - repeated rejects mute the system
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------- vocabulary
WAKE = 'hey_machina'
UNKNOWN = '_unknown'
SILENCE = '_silence'

COMMANDS = ['start', 'halt', 'pause', 'resume',
            'faster', 'slower', 'confirm', 'cancel']

# commands that begin or increase motion -> require CONFIRM
REQUIRES_CONFIRM = {'start', 'resume'}
# commands refused while the guard door reads open
BLOCKED_BY_INTERLOCK = {'start', 'resume', 'faster'}


class State(Enum):
    IDLE = 'IDLE'
    LISTENING = 'LISTENING'
    AWAIT_CONFIRM = 'AWAIT_CONFIRM'
    COOLDOWN = 'COOLDOWN'
    LOCKOUT = 'LOCKOUT'


class Tone(Enum):
    ACCEPT = 'accept'      # rising
    REJECT = 'reject'      # falling
    PROMPT = 'prompt'      # double, asking for confirmation
    BOUNDARY = 'boundary'  # speed limit reached
    LOCKOUT = 'lockout'    # triple


@dataclass
class Config:
    conf_threshold: float = 0.85
    vote_n: int = 3
    vote_m: int = 5
    cmd_window_ms: int = 3000
    confirm_timeout_ms: int = 3000
    refractory_ms: int = 1000
    max_rejects: int = 3
    cooldown_ms: int = 10000
    speed_min: int = 1
    speed_max: int = 5
    speed_default: int = 3


@dataclass
class Machine:
    """Mirror of the physical outputs."""
    motor_on: bool = False
    speed: int = 3
    paused_speed: int = 0          # speed retained across PAUSE
    relay_enable: bool = False
    relay_direction: bool = False

    def duty(self):
        table = {0: 0, 1: 90, 2: 130, 3: 165, 4: 205, 5: 255}
        return table[self.speed] if self.motor_on else 0


@dataclass
class Event:
    kind: str          # 'accepted' | 'rejected' | 'prompt' | 'state' | 'safety'
    label: str = ''
    message: str = ''
    tone: Tone = None
    confidence: float = 0.0
    outcome: str = ''  # audit field


@dataclass
class IntentFSM:
    cfg: Config = field(default_factory=Config)

    def __post_init__(self):
        self.state = State.IDLE
        self.machine = Machine(speed=self.cfg.speed_default)
        self.votes = deque(maxlen=self.cfg.vote_m)
        self.estop_tripped = False
        self.interlock_open = False
        self.consecutive_rejects = 0
        self.pending = None            # command awaiting CONFIRM
        self._state_entered_ms = 0
        self._refractory_until_ms = 0
        self.log = []                  # audit trail

    # ------------------------------------------------------------ helpers
    def _goto(self, state, now_ms):
        self.state = state
        self._state_entered_ms = now_ms

    def _emit(self, ev, now_ms):
        ev_dict = {'t': now_ms, 'state': self.state.value, 'kind': ev.kind,
                   'label': ev.label, 'confidence': round(ev.confidence, 3),
                   'outcome': ev.outcome or ev.kind, 'message': ev.message}
        self.log.append(ev_dict)
        return ev

    def _reject(self, now_ms, label, conf, reason, message):
        self.consecutive_rejects += 1
        ev = Event('rejected', label, message, Tone.REJECT, conf, reason)
        self._emit(ev, now_ms)
        if self.consecutive_rejects >= self.cfg.max_rejects:
            self.consecutive_rejects = 0
            self._goto(State.COOLDOWN, now_ms)
            self.votes.clear()
            return [ev, self._emit(Event('state', '', 'Too many rejects - cooling down',
                                         Tone.LOCKOUT, 0.0, 'cooldown'), now_ms)]
        self._goto(State.IDLE, now_ms)
        self.votes.clear()
        return [ev]

    def _accept(self, now_ms, label, conf, message, tone=Tone.ACCEPT):
        self.consecutive_rejects = 0
        self._refractory_until_ms = now_ms + self.cfg.refractory_ms
        self.votes.clear()
        return [self._emit(Event('accepted', label, message, tone, conf, 'accepted'), now_ms)]

    # ------------------------------------------------------------ safety inputs
    def set_estop(self, tripped, now_ms=0):
        """Hardwired E-stop. Firmware cannot override this."""
        was = self.estop_tripped
        self.estop_tripped = tripped
        if tripped and not was:
            self.machine.motor_on = False
            self.machine.relay_enable = False
            self.pending = None
            self.votes.clear()
            self._goto(State.LOCKOUT, now_ms)
            return [self._emit(Event('safety', 'E-STOP', 'EMERGENCY STOP - all outputs de-energised',
                                     Tone.LOCKOUT, 1.0, 'lockout'), now_ms)]
        return []

    def reset(self, now_ms=0):
        """Manual reset. Only clears lockout if the E-stop has been released."""
        if self.estop_tripped:
            return [self._emit(Event('safety', '', 'Reset refused - E-stop still engaged',
                                     Tone.REJECT, 0.0, 'reset_refused'), now_ms)]
        if self.state is State.LOCKOUT:
            self._goto(State.IDLE, now_ms)
            self.consecutive_rejects = 0
            return [self._emit(Event('state', '', 'Reset - system armed',
                                     Tone.ACCEPT, 0.0, 'reset'), now_ms)]
        return []

    def set_interlock(self, is_open, now_ms=0):
        """Guard door. Opening it stops motion immediately."""
        was = self.interlock_open
        self.interlock_open = is_open
        if is_open and not was and self.machine.motor_on:
            self.machine.motor_on = False
            self.machine.relay_enable = False
            return [self._emit(Event('safety', 'INTERLOCK', 'Guard opened - motion stopped',
                                     Tone.REJECT, 1.0, 'interlock_stop'), now_ms)]
        return []

    # ------------------------------------------------------------ time
    def tick(self, now_ms):
        """Call regularly. Handles every timeout."""
        out = []
        dwell = now_ms - self._state_entered_ms
        if self.state is State.LISTENING and dwell > self.cfg.cmd_window_ms:
            self._goto(State.IDLE, now_ms)
            self.votes.clear()
            out.append(self._emit(Event('state', '', 'Listening window closed',
                                        None, 0.0, 'window_timeout'), now_ms))
        elif self.state is State.AWAIT_CONFIRM and dwell > self.cfg.confirm_timeout_ms:
            lbl = self.pending or ''
            self.pending = None
            self._goto(State.IDLE, now_ms)
            self.votes.clear()
            out.append(self._emit(Event('rejected', lbl, 'No confirmation - command discarded',
                                        Tone.REJECT, 0.0, 'confirm_timeout'), now_ms))
        elif self.state is State.COOLDOWN and dwell > self.cfg.cooldown_ms:
            self._goto(State.IDLE, now_ms)
            out.append(self._emit(Event('state', '', 'Cooldown over - listening again',
                                        None, 0.0, 'cooldown_end'), now_ms))
        return out

    # ------------------------------------------------------------ main input
    def feed(self, label, confidence, now_ms):
        """One inference result from the classifier."""
        out = self.tick(now_ms)

        # layer 7 / safety: nothing is heard in these states
        if self.state is State.LOCKOUT:
            return out
        if self.state is State.COOLDOWN:
            return out

        # layer 6: refractory period after any actuation
        if now_ms < self._refractory_until_ms:
            return out

        # non-speech never counts
        if label in (SILENCE, UNKNOWN):
            self.votes.append(None)
            return out

        # layer 2: confidence threshold
        if confidence < self.cfg.conf_threshold:
            self.votes.append(None)
            return out

        # layer 3: N-of-M voting
        self.votes.append(label)
        if self.votes.count(label) < self.cfg.vote_n:
            return out

        return out + self._dispatch(label, confidence, now_ms)

    # ------------------------------------------------------------ dispatch
    def _dispatch(self, label, conf, now_ms):
        # layer 1: wake-word gate
        if self.state is State.IDLE:
            if label == WAKE:
                self._goto(State.LISTENING, now_ms)
                self.votes.clear()
                return [self._emit(Event('state', WAKE, 'Listening...',
                                         Tone.PROMPT, conf, 'wake'), now_ms)]
            return []   # commands are simply not heard before the wake word

        if self.state is State.LISTENING:
            if label == WAKE:
                self._state_entered_ms = now_ms   # re-arm the window
                self.votes.clear()
                return []
            if label in ('confirm', 'cancel'):
                return self._reject(now_ms, label, conf, 'nothing_pending',
                                    'Nothing to confirm')
            return self._execute_or_prompt(label, conf, now_ms)

        if self.state is State.AWAIT_CONFIRM:
            if label == 'confirm':
                cmd = self.pending
                self.pending = None
                # layer 5: interlock re-checked at the moment of actuation
                if cmd in BLOCKED_BY_INTERLOCK and self.interlock_open:
                    return self._reject(now_ms, cmd, conf, 'rejected_interlock',
                                        'Guard door open - refused')
                self._goto(State.IDLE, now_ms)
                return self._apply(cmd, conf, now_ms)
            if label == 'cancel':
                lbl = self.pending
                self.pending = None
                self._goto(State.IDLE, now_ms)
                self.votes.clear()
                self.consecutive_rejects = 0
                return [self._emit(Event('rejected', lbl, 'Cancelled',
                                         Tone.REJECT, conf, 'cancelled'), now_ms)]
            return self._reject(now_ms, label, conf, 'rejected_not_confirm',
                                'Expected CONFIRM or CANCEL')
        return []

    def _execute_or_prompt(self, cmd, conf, now_ms):
        # layer 5: interlock
        if cmd in BLOCKED_BY_INTERLOCK and self.interlock_open:
            return self._reject(now_ms, cmd, conf, 'rejected_interlock',
                                'Guard door open - refused')
        # layer 4: confirm handshake for motion-initiating commands
        if cmd in REQUIRES_CONFIRM:
            self.pending = cmd
            self._goto(State.AWAIT_CONFIRM, now_ms)
            self.votes.clear()
            return [self._emit(Event('prompt', cmd, f'{cmd.upper()} - say CONFIRM',
                                     Tone.PROMPT, conf, 'awaiting_confirm'), now_ms)]
        self._goto(State.IDLE, now_ms)
        return self._apply(cmd, conf, now_ms)

    def _apply(self, cmd, conf, now_ms):
        m, c = self.machine, self.cfg
        if cmd == 'start':
            m.motor_on = True
            m.relay_enable = True
            return self._accept(now_ms, cmd, conf, f'Motor running at speed {m.speed}')
        if cmd == 'halt':
            m.motor_on = False
            m.relay_enable = False
            m.paused_speed = 0
            m.speed = c.speed_default
            return self._accept(now_ms, cmd, conf, 'Motor halted, state cleared')
        if cmd == 'pause':
            m.paused_speed = m.speed if m.motor_on else 0
            m.motor_on = False
            m.relay_enable = False
            return self._accept(now_ms, cmd, conf, 'Paused, speed retained')
        if cmd == 'resume':
            if m.paused_speed:
                m.speed = m.paused_speed
            m.motor_on = True
            m.relay_enable = True
            return self._accept(now_ms, cmd, conf, f'Resumed at speed {m.speed}')
        if cmd == 'faster':
            if m.speed >= c.speed_max:
                return self._accept(now_ms, cmd, conf,
                                    f'Already at maximum speed {c.speed_max}', Tone.BOUNDARY)
            m.speed += 1
            return self._accept(now_ms, cmd, conf, f'Speed {m.speed}')
        if cmd == 'slower':
            if m.speed <= c.speed_min:
                return self._accept(now_ms, cmd, conf,
                                    f'Already at minimum speed {c.speed_min}', Tone.BOUNDARY)
            m.speed -= 1
            return self._accept(now_ms, cmd, conf, f'Speed {m.speed}')
        return []

    # ------------------------------------------------------------ view
    def status(self):
        return {
            'state': self.state.value,
            'motor_on': self.machine.motor_on,
            'speed': self.machine.speed,
            'duty': self.machine.duty(),
            'relay_enable': self.machine.relay_enable,
            'estop': self.estop_tripped,
            'interlock_open': self.interlock_open,
            'pending': self.pending,
            'rejects': self.consecutive_rejects,
        }
