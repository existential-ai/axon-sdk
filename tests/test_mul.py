import pytest

from axon_sdk.networks import MultiplierNetwork
from axon_sdk.primitives import DataEncoder
from axon_sdk.simulator import Simulator


@pytest.mark.parametrize(
    "val1, val2",
    [
        (0.5, 0.5),
        (1.0, 1.0),
        (0.1, 1.0),
        (1.0, 0.5),
        (0.5, 0.45),
        (0.3, 0.9),
        (0.11, 0.11),
        (0.01, 0.99),
        (0.12, 0.33),
        (0.42, 0.42),
    ],
)
def test_mul(val1, val2):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)

    net = MultiplierNetwork(encoder)
    sim = Simulator(net, encoder)
    sim.apply_input_value(val1, neuron=net.input1, t0=0)
    sim.apply_input_value(val2, neuron=net.input2, t0=0)
    sim.simulate(400)

    spikes = sim.spike_log.get(net.output.uid, [])

    assert len(spikes) == 2, f"Expected 2 spikes, got {len(spikes)}"

    interval = spikes[1] - spikes[0]
    decoded_value = encoder.decode_interval(interval)
    actual_value = val1 * val2

    assert actual_value == pytest.approx(
        decoded_value, abs=1e-2
    ), f"Expected decoded value {actual_value}, got {decoded_value}"


@pytest.mark.parametrize(
    "val1, val2",
    [
        (0.0, 0.0),
    ],
)
def test_mul_zero(val1, val2):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)

    net = MultiplierNetwork(encoder)
    sim = Simulator(net, encoder)
    sim.apply_input_value(val1, neuron=net.input1, t0=0)
    sim.apply_input_value(val2, neuron=net.input2, t0=0)
    sim.simulate(400)

    spikes = sim.spike_log.get(net.output.uid, [])

    # Network not working when both inputs are 0 since output never reaches threshold
    assert len(spikes) == 0, f"Expected 0 spikes, got {len(spikes)}"
