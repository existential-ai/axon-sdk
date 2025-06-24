import pytest

from axon_sdk.networks import SubtractorNetwork
from axon_sdk.primitives import DataEncoder
from axon_sdk.simulator import Simulator


def subtract(encoder, val1, val2) -> float:
    net = SubtractorNetwork(encoder, module_name="sub")
    sim = Simulator(net, encoder, dt=0.01)
    sim.apply_input_value(val1, net.input1, t0=0)
    sim.apply_input_value(val2, net.input2, t0=0)
    sim.simulate(300)
    output_plus = sim.spike_log.get(net.output_plus.uid, [])
    output_minus = sim.spike_log.get(net.output_minus.uid, [])
    return output_plus, output_minus


@pytest.mark.parametrize(
    "val1, val2, result",
    [
        (0.5, 0.4, 0.1),
        (0.9, 0.1, 0.8),
        (1.0, 0.5, 0.5),
        (0.5, 0.45, 0.05),
        (1.0, 1.0, 0.0),
        (0.45, 0.43, 0.02),
        (1.0, 0.99, 0.01),
        (0.01, 0.01, 0.0),
        (0.0002, 0.0001, 0.0001),
        (0.0052, 0.0001, 0.0051),
        (0.0522, 0.0011, 0.0511),
        (0.5222, 0.0111, 0.5111),
        (0.9001, 0.0001, 0.9000),
        (0.9999, 0.0001, 0.9998),
        (0.0002, 0.0001, 0.0001),
        (0.0001, 0.0001, 0.000),
    ],
)
def test_positive_subtraction(val1, val2, result):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)

    plus_log, minus_log = subtract(encoder, val1, val2)
    print(len(plus_log))

    assert len(minus_log) == 0, "Expected no spikes in negative output"
    assert len(plus_log) == 2, "Expected 2 spikes in positive output"

    spike1 = plus_log[0]
    spike2 = plus_log[1]

    decoded_value = encoder.decode_interval(spike2 - spike1)

    assert result == pytest.approx(
        decoded_value, abs=1e-3
    ), f"Expected decoded value {result}, got {decoded_value}"



@pytest.mark.parametrize(
    "val1, val2, result",
    [
        (0.2, 0.4, -0.2),
        (0.1, 0.5, -0.4),
        (0.5, 0.7, -0.2),
        (0.0, 1.0, -1.0),
        (0.1, 1.0, -0.9),
        (0.05, 0.1, -0.05),
        (0.93, 1.0, -0.07),
        (0.86, 0.9, -0.04),
        (0.0001, 0.0002, -0.0001),
        (0.0001, 0.0052, -0.0051),
        (0.0011, 0.0522, -0.0511),
        (0.0111, 0.5222, -0.5111),
        (0.0001, 0.9001, -0.9000),
        (0.0001, 0.9999, -0.9998),
        (0.0001, 0.0002, -0.0001),
    ],
)
def test_negative_subtraction(val1, val2, result):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)

    plus_log, minus_log = subtract(encoder, val1, val2)
    print(len(plus_log))

    assert len(minus_log) == 2, f"Expected 2 spikes in negative output, got {len(minus_log)}"
    assert len(plus_log) == 0, f"Expected no spikes in positive output, got {len(plus_log)}"

    spike1 = minus_log[0]
    spike2 = minus_log[1]

    decoded_value = encoder.decode_interval(spike2 - spike1)

    assert -1 * result == pytest.approx(
        decoded_value, abs=1e-3
    ), f"Expected decoded value {result}, got {decoded_value}"



@pytest.mark.parametrize(
    "val1, val2, result",
    [
        (0.2, 0.2, 0.0),
        (0.5, 0.5, 0.0),
        (0.9, 0.9, 0.0),
        (1.0, 1.0, 0.0),
        (0.45, 0.45, 0.0),
        (0.99, 0.99, 0.0),
        (0.01, 0.01, 0.0),
        (0.12, 0.12, 0.0),
        (0.47, 0.47, 0.0),
        (0.2, 0.2, 0.0),
        
    ],
)
def test_zero_subtraction(val1, val2, result):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)

    plus_log, minus_log = subtract(encoder, val1, val2)
    print(len(plus_log))

    assert len(minus_log) == 0, f"Expected no spikes in negative output, got {len(minus_log)}"
    assert len(plus_log) == 2, f"Expected 2 spikes in positive output, got {len(plus_log)}"

    spike1 = plus_log[0]
    spike2 = plus_log[1]

    decoded_value = encoder.decode_interval(spike2 - spike1)

    assert -1 * result == pytest.approx(
        decoded_value, abs=1e-3
    ), f"Expected decoded value {result}, got {decoded_value}"
