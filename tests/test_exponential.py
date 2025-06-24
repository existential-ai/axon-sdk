import pytest
from axon_sdk.networks import ExponentialNetwork
from axon_sdk.primitives import DataEncoder
from axon_sdk import Simulator
import math


# Formulas from STICK paper
def expected_exp_output_delay(x, encoder: DataEncoder, tf):
    try:
        delay = encoder.Tcod * math.exp(-x * encoder.Tcod / tf)
        Tout = encoder.Tmin + delay
        return Tout
    except:
        return float("nan")


def decode_exponential(output_interval, encoder: DataEncoder, tf):
    return ((output_interval - encoder.Tmin) / encoder.Tcod) ** (-tf / encoder.Tcod)


@pytest.mark.parametrize(
    "input_value",
    [
        (0.5),
        (0.1),
        (0.5),
        (0.9),
        (1.0),
        (0.05),
        (0.23),
        (0.34),
        (0.45),
        (0.54),
        (0.67),
        (0.72),
        (0.81),
        (0.92),
    ],
)
def test_log_output_delay(input_value):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = ExponentialNetwork(encoder)

    sim = Simulator(net, encoder, dt=0.01)
    sim.apply_input_value(input_value, neuron=net.input, t0=0)
    sim.simulate(300)

    output_spikes = sim.spike_log.get(net.output.uid, [])

    assert (
        len(output_spikes) == 2
    ), f"Expected 2 output spikes, got {len(output_spikes)}"

    output_interval = output_spikes[1] - output_spikes[0]

    expected_output_delay = expected_exp_output_delay(input_value, encoder, net.tf)

    assert (
        pytest.approx(expected_output_delay, abs=1e-1) == output_interval
    ), f"Expected delay {expected_output_delay}, got {output_interval}"

    expected_output_value = encoder.decode_interval(expected_output_delay)
    decoded_value = encoder.decode_interval(output_interval)

    assert (
        pytest.approx(expected_output_value, abs=1e-3) == decoded_value
    ), f"Expected decoded value {expected_output_value}, got {decoded_value}"


@pytest.mark.parametrize(
    "Tmin, Tcod",
    [
        (5, 50),
        (20, 200),
        (10, 500),
    ],
)
@pytest.mark.parametrize(
    "input_value",
    [
        0.5,
        0.1,
        0.5,
        0.9,
        1.0,
        0.05,
        0.23,
        0.34,
        0.45,
        0.54,
        0.67,
        0.72,
        0.81,
        0.92,
    ],
)
def test_custom_encoder_parameters(
    Tmin,
    Tcod,
    input_value,
):
    encoder = DataEncoder(Tmin=Tmin, Tcod=Tcod)
    net = ExponentialNetwork(encoder)

    sim = Simulator(net, encoder, dt=0.01)
    sim.apply_input_value(input_value, neuron=net.input, t0=0)
    sim.simulate(600)

    output_spikes = sim.spike_log.get(net.output.uid, [])

    assert (
        len(output_spikes) == 2
    ), f"Expected 2 output spikes, got {len(output_spikes)}"

    output_interval = output_spikes[1] - output_spikes[0]

    expected_output_delay = expected_exp_output_delay(input_value, encoder, net.tf)

    assert (
        pytest.approx(expected_output_delay, abs=1e-1) == output_interval
    ), f"Expected delay {expected_output_delay}, got {output_interval}"

    expected_output_value = encoder.decode_interval(expected_output_delay)
    decoded_value = encoder.decode_interval(output_interval)

    assert (
        pytest.approx(expected_output_value, abs=1e-3) == decoded_value
    ), f"Expected decoded value {expected_output_value}, got {decoded_value}"
