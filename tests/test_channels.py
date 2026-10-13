"""Channel mapping per control type, using real fixtures from api.casambi.com.

One test per kind of channel layout, not per fixture id:
dimmer+rgb+whitecolorbalance+temperature, several named sliders, dimmer+sliders,
presence+lux. Fixtures in tests/fixtures/fixtures_api/<id>.json.
"""

import json
import logging
import pathlib

from CasambiBt._network import (
    _parseUnitType,
    _withoutDisabledSensors,
    _withUnitCctRange,
)
from CasambiBt._unit import Unit, UnitControlType

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "fixtures_api"


def make_unit(fixture_id: int, settings: dict | None = None) -> Unit:
    unit_type = _parseUnitType(
        json.loads((FIXTURES / f"{fixture_id}.json").read_text()),
        logging.getLogger(__name__),
    )
    unit_type = _withUnitCctRange(unit_type, settings or {})
    unit_type = _withoutDisabledSensors(unit_type, settings or {})
    return Unit(fixture_id, 1, "uuid", "addr", "name", "1", unit_type)


def pack(*fields: tuple[int, int, int], length: int) -> bytes:
    """Pack (offset, length, value) bit fields little endian."""
    val = 0
    for off, ln, v in fields:
        assert v < 2**ln
        val |= v << off
    return val.to_bytes(length, "little")


def test_rgb_tw_channels_decode_and_roundtrip():
    """dimmer, rgb (hs), whitecolorbalance and temperature each land in their own field."""
    unit = make_unit(27399, {"cct.minKelvins": 2100.0, "cct.maxKelvins": 3500.0})
    raw = pack((0, 8, 200), (8, 18, 0x2ABCD), (26, 6, 10), (32, 8, 255), length=5)
    unit.setStateFromBytes(raw)
    state = unit.state
    assert state is not None
    assert state.dimmer == 200
    assert state.white_balance == 10
    # Per-unit range: raw 255 is the cold end the app shows, not the fixture's 6500 K.
    assert state.temperature == 3500
    assert unit.getStateAsBytes(state) == raw

    unit.setStateFromBytes(pack((32, 8, 0), length=5))
    assert unit.state.temperature == 2100


def test_temperature_range_from_unit_settings():
    plain = make_unit(27399)
    tc = plain.unitType.get_control(UnitControlType.TEMPERATURE)
    assert (tc.min, tc.max) == (2700, 6500)

    led8 = make_unit(27399, {"cct.minKelvins": 2100.0, "cct.maxKelvins": 3500.0})
    tc = led8.unitType.get_control(UnitControlType.TEMPERATURE)
    assert (tc.min, tc.max) == (2100, 3500)
    # Other controls untouched, shared fixture type not mutated.
    assert plain.unitType.get_control(UnitControlType.TEMPERATURE).max == 6500
    assert len(led8.unitType.controls) == len(plain.unitType.controls)

    # A unit type without temperature keeps its identity even with cct settings
    # (Sliders/RGBW units carry cct.* in the network but have no TW channel).
    sliders = make_unit(40271)
    assert (
        _withUnitCctRange(
            sliders.unitType, {"cct.minKelvins": 1800.0, "cct.maxKelvins": 3000.0}
        )
        is sliders.unitType
    )


def test_named_slider_channels_decode_and_roundtrip():
    """4CH: every slider channel keeps its own value and its fixture name."""
    unit = make_unit(37068)
    sliders = [c for c in unit.unitType.controls if c.type == UnitControlType.SLIDER]
    assert [c.name for c in sliders] == ["CH1", "CH2", "CH3", "CH4"]

    raw = bytes([10, 20, 30, 40])
    unit.setStateFromBytes(raw)
    assert unit.state.sliders == {0: 10, 8: 20, 16: 30, 24: 40}
    # Re-encoding must not smear one channel over the others.
    assert unit.getStateAsBytes(unit.state) == raw


def test_dimmer_with_slider_channels_roundtrip():
    """Sliders/RGBW: changing the dimmer keeps Red/Green/Blue/White apart."""
    unit = make_unit(40271)
    assert [c.name for c in unit.unitType.controls[1:]] == [
        "Red",
        "Green",
        "Blue",
        "White",
    ]
    unit.setStateFromBytes(bytes([255, 1, 2, 3, 4]))
    state = unit.state
    state.dimmer = 100
    assert unit.getStateAsBytes(state) == bytes([100, 1, 2, 3, 4])


def test_presence_and_lux():
    unit = make_unit(5511)
    unit.setStateFromBytes(pack((0, 2, 1), (2, 12, 1234), length=2))
    assert unit.state.presence == 1
    assert unit.state.lux == 1234

    unit.setStateFromBytes(pack((0, 2, 0), (2, 12, 7), length=2))
    assert unit.state.presence == 0
    assert unit.state.lux == 7


def test_sensor_disabled_in_app_is_not_mapped():
    """BM Flur 1-3 have lux.disabled: their lux bits are 0, not a measurement."""
    unit = make_unit(5511, {"lux.disabled": 1.0, "presence.disabled": 0.0})
    assert unit.unitType.get_control(UnitControlType.LUX) is None
    assert unit.unitType.get_control(UnitControlType.PRESENCE) is not None

    unit.setStateFromBytes(pack((0, 2, 1), (2, 12, 0), length=2))
    assert unit.state.presence == 1
    assert unit.state.lux is None

    enabled = make_unit(5511, {"lux.disabled": 0.0})
    assert enabled.unitType.get_control(UnitControlType.LUX) is not None
