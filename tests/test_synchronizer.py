import pytest
import logging
from axon_sdk.networks import MemoryNetwork, SynchronizerNetwork
from axon_sdk.primitives import DataEncoder
from axon_sdk import Simulator

LOGGER = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "N, values",
    [
        (1, [0.5]),
        (3, [0.2, 0.4, 0.6]),
        (5, [0.1, 0.2, 0.3, 0.4, 0.5]),
    ],
)
def test_synchronizer_network_init(N, values):
    encoder = DataEncoder(Tcod=100)
    syncnet = SynchronizerNetwork(encoder, N=N)

    assert len(syncnet.input_neurons) == N
    assert len(syncnet.output_neurons) == N
    assert len(syncnet.memory_blocks) == N

    for memory in syncnet.memory_blocks:
        assert isinstance(memory, MemoryNetwork)


def test_sync_neuron_connections():
    encoder = DataEncoder(Tcod=100)
    syncnet = SynchronizerNetwork(encoder, N=2)

    for memory in syncnet.memory_blocks:
        assert len(memory.ready.out_synapses) == 1
    assert len(syncnet.sync.out_synapses) == 2


def test_input_output_timing(N=3, values=[0.1, 0.4, 0.7]):
    encoder = DataEncoder(Tcod=100)
    syncnet = SynchronizerNetwork(encoder, N=N)

    # Set up simulator
    sim = Simulator(net=syncnet, encoder=encoder, dt=0.01)

    t0 = 10  # Time to begin encoding spikes
    deltas = [0, 20, 40]

    # Apply inputs at different times
    for i, (delta, val) in enumerate(zip(deltas, values)):
        sim.apply_input_value(
            value=val, neuron=syncnet.input_neurons[i], t0=t0 + delta
        )

    # Run simulation for enough time to capture output
    sim.simulate(simulation_time=400)

    # Collect spike times to ensure synchronisation
    sync_out_spikes = []

    # Collect and decode outputs
    for i, out_neuron in enumerate(syncnet.output_neurons):
        spikes = sim.spike_log.get(out_neuron.uid, [])
        if len(spikes) >= 2:
            interval = spikes[1] - spikes[0]
            sync_out_spikes.append(spikes[0])
            decoded_value = encoder.decode_interval(interval)
            expected_value = values[i]

            assert decoded_value == pytest.approx(
                expected_value, abs=1e-2
            ), f"Expected decoded value {expected_value}, got {decoded_value}"
        else:
            pytest.fail(f"Output[{i}] missing second spike: {spikes}")

    assert len(set(sync_out_spikes)) == 1


def test_synchronizer_network_with_custom_encoder_parameters(
    N=6, values=[0.3, 0.6, 0.1, 0.5, 0.4, 0.45]
):
    custom_encoder = DataEncoder(Tmin=5, Tcod=100)
    syncnet = SynchronizerNetwork(custom_encoder, N=N)

    # Set up simulator
    sim = Simulator(net=syncnet, encoder=custom_encoder, dt=0.01)

    t0 = 5  # Time to begin encoding spikes
    deltas = [0, 8, 17, 5, 34, 23]

    # Apply inputs using simulator-provided method
    for i, (delta, val) in enumerate(zip(deltas, values)):
        sim.apply_input_value(
            value=val, neuron=syncnet.input_neurons[i], t0=t0 + delta
        )

    # Run simulation for enough time to capture output
    sim.simulate(simulation_time=400)

    # Collect spike times to ensure synchronisation
    sync_out_spikes = []

    # Collect and decode outputs
    for i, out_neuron in enumerate(syncnet.output_neurons):
        spikes = sim.spike_log.get(out_neuron.uid, [])
        if len(spikes) >= 2:
            interval = spikes[1] - spikes[0]
            sync_out_spikes.append(spikes[0])
            decoded_value = custom_encoder.decode_interval(interval)
            expected_value = values[i]

            assert decoded_value == pytest.approx(
                expected_value, abs=1e-2
            ), f"Expected decoded value {expected_value}, got {decoded_value}"
        else:
            pytest.fail(f"Output[{i}] missing second spike: {spikes}")

    assert len(set(sync_out_spikes)) == 1


def test_synchronizer_network_with_no_inputs(N=2):
    encoder = DataEncoder(Tcod=100)
    syncnet = SynchronizerNetwork(encoder, N=N)

    # Set up simulator
    sim = Simulator(net=syncnet, encoder=encoder, dt=0.01)

    # Run simulation for enough time to capture output
    sim.simulate(simulation_time=400)

    for neuron in syncnet.output_neurons:
        spikes = sim.spike_log.get(neuron.uid, [])
        assert (
            len(spikes) == 0
        ), f"Output[{neuron.name}] should not have any spikes"


if __name__ == "__main__":
    pytest.main()
