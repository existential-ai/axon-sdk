from stick_emulator.primitives import (
    DataEncoder,
    ExplicitNeuron,
    SpikingNetworkModule,
    CancelableEventQueue,
    SpikeHitEvent,
    NeuronResetEvent,
    UniqueEvent,
)

from stick_emulator.networks import InvertingMemoryNetwork

import numpy as np

from typing import Optional


class PredSimulator:
    def __init__(
        self, net: SpikingNetworkModule, encoder: DataEncoder, dt: float = 0.01
    ) -> None:
        self.net = net
        self.encoder = encoder
        self.dt = dt

        # Predictive simulation engine
        self._event_queue = CancelableEventQueue()
        self._provisional_events: dict[str, list[UniqueEvent]] = {}

        self.spike_log: dict[str, list[float]] = {}
        self.voltage_log: dict[str, list[tuple]] = {}

        for neuron in self.net.neurons:
            self.spike_log[neuron.uid] = []
            self.voltage_log[neuron.uid] = []
            self._provisional_events[neuron.uid] = []

    def apply_input_value(self, value: float, neuron: ExplicitNeuron, t0: float = 0):
        assert value >= 0.0 and value <= 1.0
        spike_interval = self.encoder.encode_value(value)
        for t_spike_in_interval in spike_interval:
            self.apply_input_spike(neuron=neuron, t=t_spike_in_interval)

    def apply_input_spike(self, neuron: ExplicitNeuron, t: float):
        # Artifically forcing a spike is done by simulating the arrival
        # of a V-type spike to the neuron
        Vt = neuron.Vt
        spike_event = SpikeHitEvent(t=t, hitNeuron=neuron, synapse_type="V", weight=Vt)
        self._event_queue.add_event(spike_event)

    def _log_spike_occurrence(self, neuron: ExplicitNeuron, t: float) -> None:
        self.spike_log[neuron.uid].append(t)

    def _cancel_provisional_events_from(self, neuron: ExplicitNeuron) -> None:
        prov_events = self._provisional_events[neuron.uid]
        for event in prov_events:
            self._event_queue.remove(event)

    def _remove_provisional_events_from(self, neuron: ExplicitNeuron) -> None:
        self._provisional_events[neuron.uid] = []

    def _add_provisional_event_to(
        self, neuron: ExplicitNeuron, event: UniqueEvent
    ) -> None:
        self._provisional_events[neuron.uid].append(event)

    def _propagate_spikes(self, t0: float, neuron: ExplicitNeuron):
        for syn in neuron.out_synapses:
            spike_event = SpikeHitEvent(
                t=t0 + syn.delay,
                hitNeuron=syn.post_neuron,
                synapse_type=syn.type,
                weight=syn.weight,
            )
            self._event_queue.add_event(spike_event)
            self._add_provisional_event_to(neuron=neuron, event=spike_event)

    def _query_reset_event(self, t0: float, neuron: ExplicitNeuron):
        reset_event = NeuronResetEvent(t=t0, resetNeuron=neuron)
        self._event_queue.add_event(reset_event)
        self._add_provisional_event_to(neuron=neuron, event=reset_event)

    def _predict_spike_steps_fixed(
        self, neuron: ExplicitNeuron, dt, max_steps=50000
    ) -> Optional[float]:
        V0 = neuron.V
        ge = neuron.ge
        gf = neuron.gf if neuron.gate else 0.0
        tm = neuron.tm
        tf = neuron.tf
        Vt = neuron.Vt
        lo, hi = 0, max_steps
        for _ in range(64):  # up to 2^16 resolution
            mid = (lo + hi) // 2
            t = mid * dt
            exp_decay = 1 - np.exp(-t / tf)
            V = V0 + (ge / tm) * t + (neuron.gate * gf * tf / tm) * exp_decay
            if V >= Vt:
                hi = mid
            else:
                lo = mid + 1
        return lo * dt if lo < max_steps else None

    def simulate(self) -> None:
        while len(self._event_queue) > 0:
            next_events = self._event_queue.pop()
            reset_events = [e for e in next_events if isinstance(e, NeuronResetEvent)]
            spike_hit_events = [e for e in next_events if isinstance(e, SpikeHitEvent)]

            for event in reset_events:
                self._remove_provisional_events_from(event.resetNeuron)
                event.resetNeuron.reset()
                # When a reset event completes it means the neuron actually spiked
                self._log_spike_occurrence(neuron=event.resetNeuron, t=event.time)

            for event in spike_hit_events:
                self._cancel_provisional_events_from(event.hitNeuron)
                event.hitNeuron.receive_synaptic_event_pred(
                    synapse_type=event.synapse_type, weight=event.weight, t0=event.time
                )
                # Might repeatedly recalculate the new spike time if several spikes
                # hit the neuron at the same timestep, but that's unlikely
                # and event if it happens, not so computationally costly
                new_spike_time = self._predict_spike_steps_fixed(
                    neuron=event.hitNeuron, dt=self.dt
                )

                if new_spike_time is not None:
                    self._query_reset_event(
                        t0=event.time + new_spike_time, neuron=event.hitNeuron
                    )
                    self._propagate_spikes(
                        t0=event.time + new_spike_time, neuron=event.hitNeuron
                    )
                else:
                    self._cancel_provisional_events_from(event.hitNeuron)


if __name__ == "__main__":
    val = 0.6
    encoder = DataEncoder()
    imn = InvertingMemoryNetwork(encoder, module_name="invmem")
    sim = PredSimulator(imn, encoder)
    sim.apply_input_value(value=val, neuron=imn.input, t0=0)
    sim.apply_input_spike(neuron=imn.recall, t=200)
    sim.simulate()

    output_spikes = sim.spike_log[imn.output.uid]
    out_val = encoder.decode_interval(output_spikes[1] - output_spikes[0])
    print(f"Input val: {val}")
    print(f"Inverted val (1-val): {out_val}")
