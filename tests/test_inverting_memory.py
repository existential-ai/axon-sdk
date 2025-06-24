import pytest
from axon_sdk.networks import InvertingMemoryNetwork
from axon_sdk.primitives import DataEncoder
from axon_sdk import Simulator


# Helper function to run encode-recall simulation
def encode_and_recall(input_value, encoder):
    net = InvertingMemoryNetwork(encoder)
    sim = Simulator(net, encoder, dt=0.01)

    sim.apply_input_value(input_value, net.input)
    sim.apply_input_spike(net.recall, t=200)
    sim.simulate(simulation_time=500)

    output_spike_log = sim.spike_log[net.output.uid]
    return output_spike_log


@pytest.mark.parametrize(
    "input_value, expected_interval",
    [
        (0.0, 110),  # Tmin + (1-0)*Tcod = 110
        (0.1, 100),  # Tmin + (1-0.1)*Tcod = 100
        (0.3, 80),  # Tmin + (1-0.3)*Tcod = 80
        (0.5, 60),  # Tmin + (1-0.5)*Tcod = 60
        (0.7, 40),  # Tmin + (1-0.7)*Tcod = 40
        (1.0, 10),  # Tmin + (1-1.0)*Tcod = 10
    ],
)
def test_inverting_memory_spike_times(input_value, expected_interval):
    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    output_spikes = encode_and_recall(input_value, encoder)

    assert (
        len(output_spikes) == 2
    ), f"Expected exactly two output spikes, got {len(output_spikes)}"

    decoded_value = encoder.decode_interval(output_spikes[1] - output_spikes[0])
    expected_value = encoder.decode_interval(expected_interval)
    assert expected_value == pytest.approx(
        decoded_value, abs=1e-2
    ), f"Expected decoded value {expected_value}, got {decoded_value}"


def test_no_spikes_for_invalid_input():
    encoder = DataEncoder()
    net = InvertingMemoryNetwork(encoder)
    with pytest.raises(Exception):
        net.apply_input_spikes(1.5)  # invalid (>1.0)


@pytest.mark.parametrize(
    "Tmin, Tcod, input_value, expected_interval",
    [
        (5, 50, 0.5, 30),  # Tmin=5, Tcod=50: 5+(1-0.5)*50=30
        (20, 200, 0.25, 170),  # Tmin=20, Tcod=200: 20+(1-0.25)*200=170
        (10, 100, 0.75, 35),  # Tmin=10, Tcod=500: 10+(1-0.75)*200=385
    ],
)
def test_custom_encoder_parameters(Tmin, Tcod, input_value, expected_interval):
    custom_encoder = DataEncoder(Tmin=Tmin, Tcod=Tcod)
    output_spikes = encode_and_recall(input_value, custom_encoder)

    expected_value = custom_encoder.decode_interval(expected_interval)
    decoded_value = custom_encoder.decode_interval(output_spikes[1] - output_spikes[0])

    assert (
        len(output_spikes) == 2
    ), "Expected exactly two output spikes with custom encoder"
    assert expected_value == pytest.approx(
        decoded_value, abs=1e-2
    ), f"Expected decoded value {input_value}, got {decoded_value}"
