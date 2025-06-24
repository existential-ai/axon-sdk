import pytest
from axon_sdk.networks import LogNetwork
from axon_sdk.primitives import DataEncoder
from axon_sdk import Simulator


def expected_log_output_delay(x: float, encoder: DataEncoder, tf: float):
    import math

    Tin = encoder.Tmin + x * encoder.Tcod
    try:
        delay = tf * math.log(encoder.Tcod / (Tin - encoder.Tmin))
        Tout = encoder.Tmin + delay
        return Tout
    except ValueError:
        return float("nan")


def decode_logarithm(output_interval: list[float], encoder: DataEncoder, tf: float):
    return (encoder.Tmin - output_interval) / tf


@pytest.mark.parametrize(
    "input_value",
    [
        (0.1),
        (0.2),
        (0.5),
        (0.9),
        (1.0),
        (0.12),
        (0.54),
        (0.63),
        (0.87),
        (0.99)
    ],
)
def test_log_output_delay(input_value: float):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = LogNetwork(encoder)

    sim = Simulator(net, encoder, dt=0.01)
    sim.apply_input_value(input_value, neuron=net.input, t0=0)
    sim.simulate(300)

    output_spikes = sim.spike_log.get(net.output.uid, [])

    assert len(output_spikes) == 2, f"Expected 2 output spikes, got {len(output_spikes)}"
    output_delay = output_spikes[1] - output_spikes[0]

    expected_output_delay = expected_log_output_delay(input_value, encoder, net.tf)
    assert expected_output_delay == pytest.approx(output_delay, abs=1e-1)

    expected_output_value = decode_logarithm(expected_output_delay, encoder, net.tf)
    decoded_value = decode_logarithm(output_delay, encoder, net.tf)

    assert expected_output_value == pytest.approx(decoded_value, abs=1e-3), f"Expected decoded value {expected_output_value}, got {decoded_value}"


@pytest.mark.parametrize(
    "Tmin, Tcod",
    [
        (5, 50,),
        (20, 200),
        (10, 200),
        (20, 300),
    ],
)
@pytest.mark.parametrize(
    "input_value",
    [
        (0.1),
        (0.2),
        (0.5),
        (0.9),
        (1.0),
        (0.12),
        (0.54),
        (0.63),
        (0.87),
        (0.99)
    ],
)
def test_custom_encoder_parameters(Tmin, Tcod, input_value):
    encoder = DataEncoder(Tmin=Tmin, Tcod=Tcod)
    net = LogNetwork(encoder)

    sim = Simulator(net, encoder, dt=0.01)
    sim.apply_input_value(input_value, neuron=net.input, t0=0)
    sim.simulate(400)

    output_spikes = sim.spike_log.get(net.output.uid, [])

    assert len(output_spikes) == 2, f"Expected 2 output spikes, got {len(output_spikes)}"
    output_delay = output_spikes[1] - output_spikes[0]

    expected_output_delay = expected_log_output_delay(input_value, encoder, net.tf)
    assert expected_output_delay == pytest.approx(output_delay, abs=1e-1)

    expected_output_value = decode_logarithm(expected_output_delay, encoder, net.tf)
    decoded_value = decode_logarithm(output_delay, encoder, net.tf)

    assert expected_output_value == pytest.approx(decoded_value, abs=1e-3), f"Expected decoded value {expected_output_value}, got {decoded_value}"

