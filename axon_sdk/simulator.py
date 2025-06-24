"""
Simulator
=========

This module defines the main `Simulator` class for executing spiking neural networks built with the STICK model.
It provides methods for input application, event propagation, spike/voltage logging, and value decoding.

Key components:
- `Simulator`: core class for managing simulation execution.
- `decode_output`: utility to read a signed value from STICK output neurons.
- `count_spikes`: utility to count total emitted spikes in a simulation.

The simulator works in discrete time with configurable timestep `dt`, executing all synaptic and neuron dynamics via
event-based updates and logging internal state.
"""

from axon_sdk.primitives import (
    SpikingNetworkModule,
    DataEncoder,
    SpikeEventQueue,
    ExplicitNeuron,
)
from axon_sdk.visualization import vis_topology, plot_chronogram
from axon_sdk.compilation import ExecutionPlan
from .compilation.compiler import OutputReader
import os

from typing import Self, Optional


class Simulator:
    """
    Core simulation engine for executing STICK spiking neural networks.

    Attributes:
        net (SpikingNetworkModule): The spiking neural network to simulate.
        encoder (DataEncoder): Object for converting values to spike intervals and back.
        dt (float): Simulation time resolution in seconds.
        spike_log (dict[str, list[float]]): Records spike times per neuron UID.
        voltage_log (dict[str, list[tuple]]): Records membrane voltages over time.
        event_queue (SpikeEventQueue): Queue of scheduled synaptic events.
    """
    def __init__(
        self, net: SpikingNetworkModule, encoder: DataEncoder, dt: float = 0.001
    ) -> None:
        """
        Initialize a Simulator instance.

        Args:
            net (SpikingNetworkModule): The spiking network to simulate.
            encoder (DataEncoder): The encoder for converting values to spike timings.
            dt (float, optional): Simulation timestep. Defaults to 1 ms.
        """
        self.net = net
        self.event_queue = SpikeEventQueue()
        self.encoder = encoder
        self.dt = dt
        self.timesteps: list[float] = []
        self.spike_log: dict[str, list[float]] = {}
        self.voltage_log: dict[str, list[tuple]] = {}
        for neuron in self.net.neurons:
            self.spike_log[neuron.uid] = []
            self.voltage_log[neuron.uid] = []

    @classmethod
    def init_with_plan(
        cls, plan: ExecutionPlan, encoder: DataEncoder, dt: float = 0.001
    ) -> Self:
        """
        Construct a simulator using an execution plan.

        Args:
            plan (ExecutionPlan): Precompiled network with input triggers.
            encoder (DataEncoder): Encoder used to encode input values.
            dt (float, optional): Timestep in seconds. Defaults to 0.001.

        Returns:
            Simulator: Initialized simulator instance.
        """
        new_instance = cls(net=plan.net, encoder=encoder, dt=dt)

        for trigger in plan.input_triggers:
            new_instance.apply_input_value(
                trigger.normalized_value, trigger.trigger_neuron
            )

        return new_instance

    def apply_input_value(self, value: float, neuron: ExplicitNeuron, t0: float = 0):
        """
        Apply a normalized value as spike interval input to a given neuron.

        Args:
            value (float): Value in [0, 1] to encode and inject.
            neuron (ExplicitNeuron): Target neuron for injection.
            t0 (float, optional): Time offset for input spike injection. Defaults to 0.
        """
        assert value >= 0.0 and value <= 1.0

        spike_interval = self.encoder.encode_value(value)
        for t_spike_in_interval in spike_interval:
            event_time = t0 + t_spike_in_interval
            self._log_spike_occurrence(neuron=neuron, t=event_time)
            for synapse in neuron.out_synapses:
                self.event_queue.add_event(
                    time=event_time + synapse.delay,
                    neuron=synapse.post_neuron,
                    synapse_type=synapse.type,
                    weight=synapse.weight,
                )

    def apply_input_spike(self, neuron: ExplicitNeuron, t: float):
        """
        Apply a single spike input to a neuron at a specified time.

        Args:
            neuron (ExplicitNeuron): Target neuron to spike.
            t (float): Time at which spike occurs.
        """
        self._log_spike_occurrence(neuron, t)
        for synapse in neuron.out_synapses:
            self.event_queue.add_event(
                time=t + synapse.delay,
                neuron=synapse.post_neuron,
                synapse_type=synapse.type,
                weight=synapse.weight,
            )

    def simulate(self, simulation_time: float):
        """
        Run the network simulation for a given total duration.

        Args:
            simulation_time (float): Total simulation duration in seconds.

        Logs:
            - Spike times in `self.spike_log`
            - Voltage traces in `self.voltage_log`
        """
        num_steps = int(simulation_time / self.dt)
        self.timesteps = [(i + 1) * self.dt for i in range(num_steps)]
        # Set to track neurons with non-zero ge, gf, or gate at the end of a timestep
        active_state_neurons: set[ExplicitNeuron] = set()

        for i, t in enumerate(self.timesteps):
            events = self.event_queue.pop_events(t)

            currently_affected_neurons = set()
            for event in events:
                # Apply synaptic event, modifying the neuron's V, ge, gf, or gate
                event.affected_neuron.receive_synaptic_event(
                    event.synapse_type, event.weight
                )
                currently_affected_neurons.add(event.affected_neuron)

            # Collect all neurons that should be simulated
            neurons_to_simulate = currently_affected_neurons.union(active_state_neurons)
            # Prepare a set to hold the neurons turning active after this `dt`
            newly_active_state_neurons = set()

            for neuron in neurons_to_simulate:
                (V_after_update, spike) = neuron.update_and_spike(self.dt)
                # neuron.V is now V_after_update
                self._log_voltage_value(neuron=neuron, V=neuron.V, timestep=i)

                if spike:
                    self._log_spike_occurrence(neuron=neuron, t=t)
                    neuron.reset()  # V becomes Vreset, ge=0, gf=0, gate=0
                    for synapse in neuron.out_synapses:
                        self.event_queue.add_event(
                            time=t + synapse.delay,
                            neuron=synapse.post_neuron,
                            synapse_type=synapse.type,
                            weight=synapse.weight,
                        )

                # After update and potential reset, check if it remains internally active for the next step
                if neuron.ge != 0.0 or neuron.gf != 0.0 or neuron.gate != 0:
                    newly_active_state_neurons.add(neuron)

            active_state_neurons = newly_active_state_neurons

        if os.getenv("VIS", "0") == "1":
            self.launch_visualization()

    def _log_spike_occurrence(self, neuron: ExplicitNeuron, t: float) -> None:
        """
        Internal method to record a spike event for a neuron.

        Args:
            neuron (ExplicitNeuron): Neuron that spiked.
            t (float): Time of spike event.
        """
        if neuron.uid in self.spike_log:
            self.spike_log[neuron.uid].append(t)
        else:
            self.spike_log[neuron.uid] = [t]

    def _log_voltage_value(
        self, neuron: ExplicitNeuron, V: float, timestep: float
    ) -> None:
        """
        Internal method to log the voltage of a neuron at a given timestep.

        Args:
            neuron (ExplicitNeuron): The neuron to log.
            V (float): Membrane voltage.
            timestep (float): Simulation step index.
        """
        self.voltage_log[neuron.uid].append((V, timestep))

    def launch_visualization(self):
        """
        Launch interactive topology and chronogram visualizations of the simulation.

        Requires `VIS=1` in environment variables.
        """
        vis_topology(self.net)
        plot_chronogram(
            timesteps=self.timesteps,
            voltage_log=self.voltage_log,
            spike_log=self.spike_log,
        )


def decode_output(sim: Simulator, reader: OutputReader) -> Optional[float]:
    """
    Decode the final signed output value from two STICK neurons after simulation.

    Args:
        sim (Simulator): Simulator instance with spike log.
        reader (OutputReader): Decoder object with read neuron handles and normalization.

    Returns:
        Optional[float]: The decoded signed value, or `None` if no output was produced.

    Raises:
        ValueError: If more than 2 spikes or invalid combinations are detected.
    """
    spikes_plus = sim.spike_log.get(reader.read_neuron_plus.uid, [])
    spikes_minus = sim.spike_log.get(reader.read_neuron_minus.uid, [])

    decoded_value = None

    if len(spikes_plus) > 0 and len(spikes_minus) > 0:
        raise ValueError("Wrong state: produced spikes in '+' and '-' neurons")
    if len(spikes_plus) and len(spikes_plus) == 2:
        intv = spikes_plus[1] - spikes_plus[0]
        decoded_value = reader.normalization * sim.encoder.decode_interval(intv)
    elif len(spikes_plus) and len(spikes_plus) != 2:
        raise ValueError("Wrong state: neuron '+' received more than 2 spikes")
    elif len(spikes_minus) and len(spikes_minus) == 2:
        intv = spikes_minus[1] - spikes_minus[0]
        decoded_value = -1 * reader.normalization * sim.encoder.decode_interval(intv)
    elif len(spikes_minus) and len(spikes_minus) != 2:
        raise ValueError("Wrong state: neuron '-' received more than 2 spikes")

    return decoded_value


def count_spikes(sim: Simulator) -> int:
    """
    Count the total number of spikes emitted by all neurons in a simulation.

    Args:
        sim (Simulator): Simulator instance.

    Returns:
        int: Total number of spikes across all neurons.
    """
    count = 0
    for neuron_spikes in sim.spike_log.values():
        count += len(neuron_spikes)
    return count
