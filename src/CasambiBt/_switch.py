import logging
from dataclasses import dataclass
from enum import Enum, unique
from time import monotonic
from typing import Final

from ._invocation import parse_invocation_stream

_LOGGER = logging.getLogger(__name__)

_BUTTON_EVENT_MIN: Final[int] = 29  # FunctionButtonEvent0
_BUTTON_EVENT_MAX: Final[int] = 36  # FunctionButtonEvent7
_INPUT_EVENT_MIN: Final[int] = 64  # FunctionNotifyInput0
_INPUT_EVENT_MAX: Final[int] = 71  # FunctionNotifyInput7

_TARGET_TYPE_BUTTON: Final[int] = 0x06
_TARGET_TYPE_INPUT: Final[int] = 0x12

_DEDUP_WINDOW_SECONDS: Final[float] = 0.5


@unique
class ButtonEventType(Enum):
    PRESS = 0x01
    RELEASE = 0x02
    HOLD = 0x09
    RELEASE_AFTER_HOLD = 0x0C
    UNKNOWN = 0xFFFF


@dataclass(frozen=True, repr=True)
class SwitchEvent:
    button_event_index: int  # 0-based index from protocol (opcode - base)
    button: int  # 1-based label = button_event_index + 1
    unit_id: int
    target_type: int  # 0x06 = button stream, 0x12 = input stream
    event: ButtonEventType
    flags: int
    extra_data: bytes


_BUTTON_DOWN_BIT: Final[int] = 0x80  # 0x06 payload[0]: 0xF9 = down, 0x79 = up
_AGE_UNIT_SECONDS: Final[float] = 0.01  # INVOCATION age counts 10 ms ticks

_INPUT_CODES: Final[frozenset[int]] = frozenset({0x01, 0x02, 0x09, 0x0C})

_UP: Final[str] = "up"
_DOWN: Final[str] = "down"
_HELD: Final[str] = "held"


class SwitchEventDecoder:
    """Stateful decoder for wireless switch events (e.g. EnOcean PTM215B).

    Each physical edge arrives on two streams, each retransmitted several times:
      - 0x06 (button stream): payload[0] bit 7 is the contact state
        (0xF9 = down, 0x79 = up).  `origin` is (unit_id << 8) | telegram
        counter; the counter steps by one per edge, so its bits carry no
        press/release meaning.
      - 0x12 (input stream): payload[0] is the node's classification, sent
        once per action: 0x02 = released after a short press, 0x09 = still
        held (~0.7 s after down), 0x0C = released after hold.  Code 0x01 was
        never observed on PTM215B.

    Measured on PTM215B (Terrasse/Kueche Wand, 2026-10-13): every down/up
    edge shows 0xF9/0x79 on 0x06, while single frames of either stream are
    regularly lost.  Therefore:
      - PRESS comes from 0x06 down.
      - A short-press release comes from whichever arrives first, 0x06 up or
        0x12 0x02; a release after hold from 0x06 up or 0x12 0x0C.
      - HOLD comes from 0x12 0x09.
    A per-button phase (up/down/held) decides what a release edge means and
    drops the second source of the same release.

    Retransmit deduplication: all copies of a frame share `origin`, and `age`
    tells how long ago (in 10 ms ticks) the frame was originated.  Copies
    arrive up to ~2 s late, so the origin time (now - age) is compared rather
    than the arrival time.
    """

    def __init__(self) -> None:
        # (unit_id, button_index, target_type, origin) -> estimated origin time
        self._seen_origins: dict[tuple[int, int, int, int], float] = {}
        # (unit_id, button_index) -> (_UP / _DOWN / _HELD, origin time of press)
        self._phase: dict[tuple[int, int], tuple[str, float]] = {}
        self._logger = _LOGGER

    def reset(self) -> None:
        """Clear all cached state (call on reconnect)."""
        self._seen_origins.clear()
        self._phase.clear()

    def _origin_time(self, age: int) -> float:
        return monotonic() - age * _AGE_UNIT_SECONDS

    def _is_retransmit(
        self,
        unit_id: int,
        button_index: int,
        target_type: int,
        origin: int,
        origin_time: float,
    ) -> bool:
        """Return True if this frame is a copy of an already accepted frame.

        Records the frame otherwise.  Copies of one frame agree on their
        origin time within a few ticks; the window guards against the origin
        counter wrapping around.
        """
        key = (unit_id, button_index, target_type, origin)
        seen = self._seen_origins.get(key)
        if seen is not None and abs(origin_time - seen) < _DEDUP_WINDOW_SECONDS:
            return True
        self._seen_origins[key] = origin_time
        return False

    def _classify(
        self,
        key: tuple[int, int],
        target_type: int,
        code: int,
        is_down: bool,
        origin_time: float,
    ) -> list[ButtonEventType]:
        """Map one accepted frame to events (usually zero or one)."""
        phase, press_time = self._phase.get(key, (_UP, 0.0))

        if target_type == _TARGET_TYPE_BUTTON and is_down:
            events = []
            if phase != _UP:
                # A new down edge proves the previous release, whose frames
                # were all lost (or are still in flight and will be stale).
                events.append(
                    ButtonEventType.RELEASE_AFTER_HOLD
                    if phase == _HELD
                    else ButtonEventType.RELEASE
                )
            self._phase[key] = (_DOWN, origin_time)
            return events + [ButtonEventType.PRESS]

        if target_type == _TARGET_TYPE_INPUT and code not in _INPUT_CODES:
            return [ButtonEventType.UNKNOWN]

        if phase == _UP or origin_time < press_time:
            # Already released, press lost, or a late frame of the previous
            # action (originated before the current press).
            return []

        if target_type == _TARGET_TYPE_BUTTON:  # up edge
            event = (
                ButtonEventType.RELEASE_AFTER_HOLD
                if phase == _HELD
                else ButtonEventType.RELEASE
            )
        elif code == ButtonEventType.RELEASE.value and phase == _DOWN:
            event = ButtonEventType.RELEASE
        elif code == ButtonEventType.RELEASE_AFTER_HOLD.value:
            event = ButtonEventType.RELEASE_AFTER_HOLD
        elif code == ButtonEventType.HOLD.value and phase == _DOWN:
            self._phase[key] = (_HELD, press_time)
            return [ButtonEventType.HOLD]
        else:
            return []  # 0x06 is the source for PRESS; others out of phase

        self._phase[key] = (_UP, press_time)
        return [event]

    def decode(self, data: bytes, packet_seq: int) -> list[SwitchEvent]:
        """Parse decrypted type-7 packet payload and return switch events."""

        frames = parse_invocation_stream(data)
        events: list[SwitchEvent] = []

        for frame in frames:
            target_type = frame.target & 0xFF
            unit_id = frame.target >> 8

            if (
                target_type == _TARGET_TYPE_BUTTON
                and _BUTTON_EVENT_MIN <= frame.opcode <= _BUTTON_EVENT_MAX
            ):
                button_event_index = frame.opcode - _BUTTON_EVENT_MIN
                code = 0
                is_down = bool(frame.payload) and bool(
                    frame.payload[0] & _BUTTON_DOWN_BIT
                )
                extra_data = frame.payload
            elif (
                target_type == _TARGET_TYPE_INPUT
                and _INPUT_EVENT_MIN <= frame.opcode <= _INPUT_EVENT_MAX
            ):
                if not frame.payload:
                    self._logger.debug(
                        "Input stream frame with empty payload, skipping."
                    )
                    continue
                button_event_index = frame.opcode - _INPUT_EVENT_MIN
                code = frame.payload[0]
                is_down = False
                extra_data = frame.payload[1:]
            else:
                self._logger.debug(
                    "Ignoring INVOCATION frame: opcode=0x%02x target_type=0x%02x.",
                    frame.opcode,
                    target_type,
                )
                continue

            origin_time = self._origin_time(frame.age)
            if self._is_retransmit(
                unit_id, button_event_index, target_type, frame.origin, origin_time
            ):
                self._logger.debug(
                    "Suppressed retransmit (0x%02x): unit_id=%d button_index=%d origin=0x%04x age=%d",
                    target_type,
                    unit_id,
                    button_event_index,
                    frame.origin,
                    frame.age,
                )
                continue

            classified = self._classify(
                (unit_id, button_event_index), target_type, code, is_down, origin_time
            )
            if not classified:
                self._logger.debug(
                    "No event for 0x%02x frame (code=0x%02x down=%s): unit_id=%d button_index=%d",
                    target_type,
                    code,
                    is_down,
                    unit_id,
                    button_event_index,
                )
            for event in classified:
                if event is ButtonEventType.UNKNOWN:
                    self._logger.debug(
                        "Unknown input event code 0x%02x in input stream frame.", code
                    )
                events.append(
                    SwitchEvent(
                        button_event_index=button_event_index,
                        button=button_event_index + 1,
                        unit_id=unit_id,
                        target_type=target_type,
                        event=event,
                        flags=frame.flags,
                        extra_data=extra_data,
                    )
                )

        if not events:
            self._logger.debug("No switch events found in packet #%s.", packet_seq)

        return events
