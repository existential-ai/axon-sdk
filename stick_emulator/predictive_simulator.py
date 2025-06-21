from stick_emulator.primitives import (
    DataEncoder,
    ExplicitNeuron,
    SpikingNetworkModule,
    CancelableEventQueue,
    SpikeHitEvent,
    NeuronResetEvent,
    UniqueEvent,
)

import numpy as np

from typing import Optional


class SpikeTime:
    def __init__(self, t: float):
        self.t = t


class SimpleNet(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, module_name: Optional[str] = None) -> None:
        super().__init__(module_name)
        self.encoder = encoder

        Vt = 10.0
        tm = 100.0
        tf = 20.0
        we = Vt
        Tsyn = 1.0

        # Create constant neuron
        self.inp = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="recall"
        )
        self.outp = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="output"
        )

        # Connect constant neuron to itself with a delay
        self.connect_neurons(self.inp, self.outp, "V", Vt, Tsyn)


class PredSimulator:
    def __init__(
        self, net: SpikingNetworkModule, encoder: DataEncoder, dt: float = 0.001
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

    def apply_input_spike(self, neuron: ExplicitNeuron, t: float):
        for synapse in neuron.out_synapses:
            new_event = SpikeHitEvent(
                t=t + synapse.delay,
                hitNeuron=synapse.post_neuron,
                synapse_type=synapse.type,
                weight=synapse.weight,
            )
            self._event_queue.add_event(new_event)

    def _cancel_provisional_events_from(self, neuron: ExplicitNeuron):
        prov_events = self._provisional_events[neuron.uid]
        for event in prov_events:
            self._event_queue.remove(event)

    def _add_provisional_events_to(self, neuron: ExplicitNeuron, event: UniqueEvent):
        prov_events = self._provisional_events[neuron.uid]
        for event in prov_events:
            self._event_queue.remove(event)

    def _propagate_spikes(self, t0: SpikeTime, neuron: ExplicitNeuron):
        for syn in neuron.out_synapses:
            spike_event = SpikeHitEvent(
                t=t0.t + syn.delay,
                hitNeuron=syn.post_neuron,
                synapse_type=syn.type,
                weight=syn.weight,
            )
            self._event_queue.add_event(spike_event)
            self._add_provisional_events_to(neuron=neuron, event=spike_event)

    def _query_reset(self, t0: SpikeTime, neuron: ExplicitNeuron):
        reset_event = NeuronResetEvent(t=t0.t, resetNeuron=neuron)
        self._event_queue.add_event(reset_event)

    def _predict_spike_steps_fixed(
        self, neuron: ExplicitNeuron, dt, max_steps=10000
    ) -> Optional[SpikeTime]:
        V0 = neuron.V
        ge = neuron.ge
        gf = neuron.gf if neuron.gate else 0.0
        tm = neuron.tm
        tf = neuron.tf
        Vt = neuron.Vt
        lo, hi = 0, max_steps
        for _ in range(32):  # up to 2^16 resolution
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
                self._cancel_provisional_events_from(event.resetNeuron)
                event.resetNeuron.reset()

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

                if new_spike_time:
                    self._query_reset(t0=new_spike_time, neuron=event.hitNeuron)
                    self._propagate_spikes(t0=new_spike_time, neuron=event.hitNeuron)



