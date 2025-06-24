import pytest
import logging
from axon_sdk.networks import ConstantNetwork
from axon_sdk.primitives import DataEncoder
from axon_sdk import Simulator

LOGGER = logging.getLogger(__name__)


# Helper function to run encode-recall simulation
def encode_and_recall(encoder, test_value):
    net = ConstantNetwork(encoder, test_value)
    sim = Simulator(net, encoder, dt=0.01)

    sim.apply_input_spike(net.recall, t=100)
    sim.simulate(simulation_time=500)

    output_spike_log = sim.spike_log[net.output.uid]
    LOGGER.info(f"Output spike log: {output_spike_log}")
    return output_spike_log


@pytest.mark.parametrize(
    "input_value, expected_interval",
    [
        (0.0, 10),  # Tmin + (0)*Tcod = 10
        (0.1, 45),  # Tmin + (0.1)*Tcod = 45
        (0.3, 115),  # Tmin + (0.3)*Tcod = 115
        (0.5, 185),  # Tmin + (0.5)*Tcod = 185
        (0.7, 255),  # Tmin + (0.7)*Tcod = 255
        (1.0, 360),  # Tmin + (1.0)*Tcod = 360
    ],
)
def test_constant_spike_times(input_value, expected_interval):
    encoder = DataEncoder(Tmin=10.0, Tcod=350.0)
    output_spikes = encode_and_recall(encoder, input_value)

    assert (
        len(output_spikes) == 2
    ), f"Expected exactly two output spikes, got {len(output_spikes)}"

    decoded_value = encoder.decode_interval(output_spikes[1] - output_spikes[0])
    expected_value = encoder.decode_interval(expected_interval)
    assert expected_value == pytest.approx(
        decoded_value, abs=1e-2
    ), f"Expected decoded value {expected_value}, got {decoded_value}"


@pytest.mark.parametrize(
    "Tmin, Tcod, input_value, expected_interval",
    [
        (5, 50, 0.5, 30),  # Tmin=5, Tcod=50: 5 + (0.5)*50 = 30
        (20, 200, 0.25, 70),  # Tmin=20, Tcod=200: 20 + (0.25)*200 = 70
        (10, 500, 0.75, 385),  # Tmin=10, Tcod=500: 10 + (0.75)*500 = 385
    ],
)
def test_custom_encoder_parameters(Tmin, Tcod, input_value, expected_interval):
    custom_encoder = DataEncoder(Tmin=Tmin, Tcod=Tcod)
    output_spikes = encode_and_recall(custom_encoder, input_value)
    print(output_spikes)

    expected_value = custom_encoder.decode_interval(expected_interval)
    decoded_value = custom_encoder.decode_interval(
        output_spikes[1] - output_spikes[0]
    )

    assert (
        len(output_spikes) == 2
    ), "Expected exactly two output spikes with custom encoder"
    assert expected_value == pytest.approx(
        decoded_value, abs=1e-2
    ), f"Expected decoded value {input_value}, got {decoded_value}"
