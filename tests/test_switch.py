"""Tests for SwitchEventDecoder — contact state, phase logic, retransmit dedup."""

import struct
from pathlib import Path

import pytest

import CasambiBt._switch as switch_module
from CasambiBt._switch import (
    _TARGET_TYPE_BUTTON,
    _TARGET_TYPE_INPUT,
    ButtonEventType,
    SwitchEventDecoder,
)

P = ButtonEventType.PRESS
R = ButtonEventType.RELEASE
H = ButtonEventType.HOLD
RAH = ButtonEventType.RELEASE_AFTER_HOLD

# ---------------------------------------------------------------------------
# Helpers to build raw INVOCATION frames
# ---------------------------------------------------------------------------
# Frame layout (from _invocation.py):
#   flags:u16 (big-endian)  — low 6 bits = payload_len
#   opcode:u8
#   origin:u16 (big-endian)
#   target:u16 (big-endian) — low byte = target_type, high byte = unit_id
#   age:u16 (big-endian)    — 10 ms ticks since the frame was originated
#   payload: payload_len bytes

_OPCODE_BUTTON_0 = 29  # FunctionButtonEvent0 → button 1
_OPCODE_INPUT_0 = 64  # FunctionNotifyInput0 → button 1
_UNIT_ID = 0x15
_DOWN = b"\xf9\x00\x00\x00"  # PTM215B 0x06 payload while pressed
_UP = b"\x79\x00\x00\x00"  # PTM215B 0x06 payload after release


def _button_frame(origin: int, payload: bytes = _DOWN, age: int = 1) -> bytes:
    """Build a 0x06 button-stream INVOCATION frame."""
    target = (_UNIT_ID << 8) | _TARGET_TYPE_BUTTON
    return (
        struct.pack(">HBHHH", len(payload), _OPCODE_BUTTON_0, origin, target, age)
        + payload
    )


def _input_frame(origin: int, event_code: int, age: int = 1) -> bytes:
    """Build a 0x12 input-stream INVOCATION frame (payload = code, 0x08)."""
    payload = bytes([event_code, 0x08])
    target = (_UNIT_ID << 8) | _TARGET_TYPE_INPUT
    return (
        struct.pack(">HBHHH", len(payload), _OPCODE_INPUT_0, origin, target, age)
        + payload
    )


class _Clock:
    """Replaces monotonic() in the decoder module."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(switch_module, "monotonic", c)
    return c


def _events(dec: SwitchEventDecoder, *frames: bytes) -> list[ButtonEventType]:
    return [e.event for f in frames for e in dec.decode(f, packet_seq=1)]


# ---------------------------------------------------------------------------
# Contact state and phase logic
# ---------------------------------------------------------------------------


class TestContactState:
    def test_down_is_press_regardless_of_origin_bits(self, clock):
        """0xF9 means pressed even when origin bit 1 is set (old decoder: RELEASE)."""
        dec = SwitchEventDecoder()
        assert _events(dec, _button_frame(0x06EF, _DOWN)) == [P]

    def test_up_after_down_is_release(self, clock):
        dec = SwitchEventDecoder()
        assert _events(
            dec, _button_frame(0x0610, _DOWN), _button_frame(0x0611, _UP)
        ) == [P, R]

    def test_short_click_release_reported_once_from_both_streams(self, clock):
        """0x06 up and 0x12 0x02 describe the same release: one RELEASE only."""
        dec = SwitchEventDecoder()
        got = _events(
            dec,
            _button_frame(0x0613, _DOWN),
            _button_frame(0x0614, _UP),
            _input_frame(0x015B, 0x02),
        )
        assert got == [P, R]

    def test_release_from_input_stream_when_up_frame_lost(self, clock):
        dec = SwitchEventDecoder()
        got = _events(dec, _button_frame(0x061B, _DOWN), _input_frame(0x0119, 0x02))
        assert got == [P, R]

    def test_long_press_gives_press_hold_release_after_hold(self, clock):
        """0x06 up after HOLD is the end of the hold, not a short release."""
        dec = SwitchEventDecoder()
        got = _events(
            dec,
            _button_frame(0x0619, _DOWN),
            _input_frame(0x0161, 0x09),
            _button_frame(0x061A, _UP),
            _input_frame(0x0118, 0x0C),
        )
        assert got == [P, H, RAH]

    def test_release_after_hold_from_input_stream_when_up_frame_lost(self, clock):
        dec = SwitchEventDecoder()
        got = _events(
            dec,
            _button_frame(0x2F87, _DOWN),
            _input_frame(0x01E2, 0x09),
            _input_frame(0x01FC, 0x0C),
        )
        assert got == [P, H, RAH]

    def test_release_without_press_is_dropped(self, clock):
        """Press frame lost: neither up nor 0x02 may produce an orphan RELEASE."""
        dec = SwitchEventDecoder()
        assert (
            _events(dec, _button_frame(0x2F7A, _UP), _input_frame(0x0113, 0x02)) == []
        )

    def test_input_stream_press_code_ignored(self, clock):
        dec = SwitchEventDecoder()
        assert _events(dec, _input_frame(0x0001, 0x01)) == []

    def test_unknown_input_code_reported(self, clock):
        dec = SwitchEventDecoder()
        assert _events(dec, _input_frame(0x0001, 0x05)) == [ButtonEventType.UNKNOWN]

    def test_button_label_unit_and_target_type(self, clock):
        dec = SwitchEventDecoder()
        (event,) = dec.decode(_button_frame(0x1000), packet_seq=1)
        assert event.button == event.button_event_index + 1 == 1
        assert event.unit_id == _UNIT_ID
        assert event.target_type == _TARGET_TYPE_BUTTON


# ---------------------------------------------------------------------------
# Retransmit deduplication
# ---------------------------------------------------------------------------


class TestDedup:
    def test_sequential_retransmits_deduplicated(self, clock):
        dec = SwitchEventDecoder()
        frame = _button_frame(0x15B0)
        assert _events(dec, frame, frame, frame) == [P]

    def test_hold_retransmits_deduplicated(self, clock):
        dec = SwitchEventDecoder()
        _events(dec, _button_frame(0x15B0))
        hold = _input_frame(0x15C0, 0x09)
        assert _events(dec, hold, hold, hold) == [H]

    def test_late_press_retransmit_after_release_suppressed(self, clock):
        dec = SwitchEventDecoder()
        press = _button_frame(0x15B0, _DOWN)
        got = _events(dec, press, _button_frame(0x15B1, _UP), press)
        assert got == [P, R]

    def test_copy_arriving_after_two_seconds_suppressed(self, clock):
        """Real case org=0x0146: copies with age 17 and 191 ticks, 1.755 s apart.

        The old 500 ms arrival window emitted RELEASE_AFTER_HOLD twice here.
        """
        dec = SwitchEventDecoder()
        _events(dec, _button_frame(0x0622, _DOWN))
        clock.now += 0.75
        _events(dec, _input_frame(0x0121, 0x09))
        clock.now += 0.82
        assert _events(dec, _input_frame(0x0146, 0x0C, age=17)) == [RAH]
        clock.now += 1.755
        assert _events(dec, _input_frame(0x0146, 0x0C, age=191)) == []

    def test_late_hold_copy_does_not_resurrect_hold(self, clock):
        """Real case org=0x0104: second HOLD copy 0.7 s later (age 5 → 72)."""
        dec = SwitchEventDecoder()
        _events(dec, _button_frame(0x0638, _DOWN))
        clock.now += 0.75
        assert _events(dec, _input_frame(0x0104, 0x09, age=5)) == [H]
        clock.now += 0.704
        assert _events(dec, _input_frame(0x0104, 0x09, age=72)) == []

    def test_same_origin_accepted_when_origin_time_differs(self, clock):
        """Counter wrap-around: same origin, but originated much later → new event."""
        dec = SwitchEventDecoder()
        frame = _button_frame(0x15B0, _DOWN)
        assert _events(dec, frame, _button_frame(0x15B1, _UP)) == [P, R]
        clock.now += 60
        assert _events(dec, frame) == [P]

    def test_second_down_edge_reports_lost_release_first(self, clock):
        """Both release frames lost: the next down edge proves the release."""
        dec = SwitchEventDecoder()
        got = _events(dec, _button_frame(0xAAA0, _DOWN), _button_frame(0xAAA2, _DOWN))
        assert got == [P, R, P]

    def test_second_down_edge_after_hold_reports_release_after_hold(self, clock):
        dec = SwitchEventDecoder()
        _events(dec, _button_frame(0xAAA0, _DOWN))
        clock.now += 0.75
        _events(dec, _input_frame(0x0001, 0x09))
        clock.now += 2
        assert _events(dec, _button_frame(0xAAA2, _DOWN)) == [RAH, P]

    def test_late_release_of_previous_click_ignored(self, clock):
        """Real case org=0x01FF: 0x02 of click 1 arrives after down of click 2.

        Its origin time lies before the second press, so it must not release
        the second press early.
        """
        dec = SwitchEventDecoder()
        _events(dec, _button_frame(0x0634, _DOWN, age=9))
        clock.now += 0.384
        got = _events(dec, _button_frame(0x0636, _DOWN, age=8))
        clock.now += 0.226
        got += _events(dec, _input_frame(0x01FF, 0x02, age=36))
        clock.now += 0.1
        got += _events(dec, _button_frame(0x0637, _UP, age=7))
        assert got == [R, P, R]

    def test_streams_do_not_share_dedup_keys(self, clock):
        """Unit 1: 0x06 origin 0x01xx can equal a 0x12 origin; both must count."""
        dec = SwitchEventDecoder()
        target_btn = (0x01 << 8) | _TARGET_TYPE_BUTTON
        target_in = (0x01 << 8) | _TARGET_TYPE_INPUT
        down = struct.pack(">HBHHH", 4, _OPCODE_BUTTON_0, 0x0119, target_btn, 1) + _DOWN
        rel = (
            struct.pack(">HBHHH", 2, _OPCODE_INPUT_0, 0x0119, target_in, 1)
            + b"\x02\x08"
        )
        assert _events(dec, down, rel) == [P, R]

    def test_late_release_of_previous_click_does_not_cut_next_hold(self, clock):
        """Late 0x02 of click 1 must not end click 2, which turns into a hold."""
        dec = SwitchEventDecoder()
        got = _events(dec, _button_frame(0x0634, _DOWN))
        clock.now += 0.4
        got += _events(dec, _button_frame(0x0636, _DOWN))
        clock.now += 0.2
        got += _events(dec, _input_frame(0x01FF, 0x02, age=40))
        clock.now += 0.6
        got += _events(dec, _input_frame(0x0104, 0x09))
        clock.now += 1.0
        got += _events(dec, _button_frame(0x0637, _UP))
        assert got == [P, R, P, H, RAH]

    def test_repeated_down_telegram_inside_hold_is_not_a_new_press(self, clock):
        """Real case Bad button 2: down 0x0986, HOLD, down 0x0987, 0x0C, up."""
        dec = SwitchEventDecoder()
        got = _events(dec, _button_frame(0x0986, _DOWN))
        clock.now += 0.7
        got += _events(dec, _input_frame(0x0170, 0x09))
        clock.now += 0.6
        got += _events(dec, _button_frame(0x0987, _DOWN))
        clock.now += 3.4
        got += _events(dec, _input_frame(0x0126, 0x0C), _button_frame(0x0988, _UP))
        assert got == [P, H, RAH]

    def test_counter_wraps_for_repeated_down(self, clock):
        dec = SwitchEventDecoder()
        got = _events(dec, _button_frame(0x09FF, _DOWN), _button_frame(0x0900, _DOWN))
        assert got == [P]


class TestReset:
    def test_reset_clears_dedup_and_phase(self, clock):
        dec = SwitchEventDecoder()
        frame = _button_frame(0x15B0, _DOWN)
        _events(dec, frame)
        dec.reset()
        assert _events(dec, frame) == [P]
        dec.reset()
        assert _events(dec, _input_frame(0x0001, 0x02)) == []  # phase back to up


# ---------------------------------------------------------------------------
# Replay of a real capture (PTM215B, 2026-10-13)
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent / "fixtures"

# Ground truth per button, from the operator's action list and the frame
# timing.  Where every frame of an edge was lost on air, the decoder cannot
# know about it; those spots are noted and expected to stay silent.
# fmt: off
_EXPECTED = {
    (6, 1): [
        P, R, P, R, P, R,  # 3 single clicks
        P, H, RAH,  # hold 5 s
        P, R, P, R,  # double click (both 0x06 up frames lost)
        P, R,  # double click, second click lost on air
        P, H, RAH,  # hold 1.4 s
        P, H, RAH,  # hold 1.7 s
        P, R,  # single click
    ],
    (6, 3): [
        P, R, P, R, P, R,  # 3 single clicks
        P, H, RAH,  # hold 3.2 s
        P, R,  # double click, second down frame lost
        P, R, P, R,  # double click
        P, H, RAH,  # hold 1.9 s
        P, H, RAH,  # hold 4.4 s
        P, H, RAH,  # hold 4.2 s
    ],
    (47, 1): [
        P, R,  # single click
        # single click, down frame lost: up + 0x02 alone stay silent
        P, R,  # single click
        P, H, RAH,  # hold 4.4 s
        P, R, P, R,  # double click
        P, R,  # double click, second click lost on air
        P, H, RAH,  # hold, 0x06 up lost
    ],
    (47, 4): [P, R],  # single click
}

_EXPECTED_BAD = {
    (9, 1): [
        P, R,  # single click
        P, H, RAH,  # hold 2.5 s
        # double click, first down frame lost: its up + 0x02 stay silent
        P, R,
    ],
    (9, 2): [
        # hold 4.7 s with a second down telegram (counter +1, no up edge)
        # 1.3 s in; the node reports one HOLD and one 0x0C
        P, H, RAH,
        P, R, P, R,  # double click
        P, H, RAH,  # hold 3.9 s
    ],
}
# fmt: on


def _replay(name: str) -> dict[tuple[int, int], list[ButtonEventType]]:
    clock = _Clock()
    dec = SwitchEventDecoder()
    got: dict[tuple[int, int], list[ButtonEventType]] = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(switch_module, "monotonic", clock)
        for line in (_FIXTURES / name).read_text().splitlines():
            if line.startswith("#"):
                continue
            ts, packet = line.split()
            h, m, s = ts.split(":")
            clock.now = int(h) * 3600 + int(m) * 60 + float(s)
            for e in dec.decode(bytes.fromhex(packet)[1:], packet_seq=0):
                got.setdefault((e.unit_id, e.button), []).append(e.event)
    return got


_CAPTURES = [
    ("ptm215b_2026-10-13.txt", _EXPECTED),
    ("ptm215b_bad_2026-10-13.txt", _EXPECTED_BAD),
]


@pytest.mark.parametrize(("name", "expected"), _CAPTURES)
def test_replay_real_capture_matches_ground_truth(name, expected):
    assert _replay(name) == expected


@pytest.mark.parametrize(("name", "expected"), _CAPTURES)
def test_replay_has_valid_sequences_without_duplicates(name, expected):
    """Every action is press→release or press→hold→release_after_hold."""
    for key, seq in _replay(name).items():
        i = 0
        while i < len(seq):
            if seq[i : i + 2] == [P, R]:
                i += 2
            elif seq[i : i + 3] == [P, H, RAH]:
                i += 3
            else:
                pytest.fail(f"{key}: invalid sequence at {i}: {seq[i : i + 3]}")
