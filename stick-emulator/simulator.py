import os
from primitives import SpikingNetworkModule, DataEncoder, SpikeEventQueue, SpikeEvent, ExplicitNeuron
from visualizer import Visualizer


class Simulator:
    def __init__(self, net:SpikingNetworkModule, encoder:DataEncoder, dt:float=0.001) -> None:
        self.net = net
        self.event_queue = SpikeEventQueue()
        self.spike_log:dict[str, list[float]] = {}
        self.voltage_log:dict[str, list[float]] = {}
        self.encoder = encoder
        self.dt = dt

    def apply_input_value(self, value:float, neuron:ExplicitNeuron, t0:float=0):
        assert value >= 0.0 and value <= 1.0
        spike_interval = self.encoder.encode_value(value)
        for t in spike_interval:
            self.log_spike(neuron=neuron, t=t0+t)
            for synapse in neuron.out_synapses:
                self.event_queue.add_event(time=t0+synapse.delay+t, neuron=synapse.post_neuron, synapse_type=synapse.type, weight=synapse.weight)

    def apply_input_spike(self, neuron:ExplicitNeuron, t:float):
        self.log_spike(neuron, t)
        for synapse in neuron.out_synapses:
            self.event_queue.add_event(time=synapse.delay+t, neuron=synapse.post_neuron, synapse_type=synapse.type, weight=synapse.weight)

    def simulate(self, simulation_time:float):
        for neuron in self.net.neurons:
            self.log_voltage(neuron=neuron, V=neuron.V)
        self.timesteps = [i * self.dt for i in range(int(simulation_time / self.dt))]
        for t in self.timesteps[1:]:
            events = self.event_queue.pop_events(t)
            for event in events:
                event.affected_neuron.receive_synaptic_event(event.synapse_type, event.weight)

            for neuron in self.net.neurons:
                (V, spike) = neuron.update_and_spike(self.dt)
                self.log_voltage(neuron=neuron, V=V)
                if spike:
                    self.log_spike(neuron=neuron, t=t)
                    neuron.reset()
                    for synapse in neuron.out_synapses:
                        self.event_queue.add_event(time=t+synapse.delay, neuron=synapse.post_neuron, synapse_type=synapse.type, weight=synapse.weight)

        if os.getenv("VIZ", "0") == "1":
            self.launch_visualization()


    def log_spike(self, neuron:ExplicitNeuron, t:float) -> None:
        if neuron.uid in self.spike_log:
            self.spike_log[neuron.uid].append(t)
        else:
            self.spike_log[neuron.uid] = [t]

    def log_voltage(self, neuron:ExplicitNeuron, V:float) -> None:
        if neuron.uid in self.voltage_log:
            self.voltage_log[neuron.uid].append(V)
        else:
            self.voltage_log[neuron.uid] = [V]

    def launch_visualization(self):
        print('Launching visualization...')
        voltage_log_with_id = {}
        spike_log_with_id = {}
        it = 0
        for neuron in self.net.neurons:
            new_id = f"{it:02}_" + neuron.uid
            voltage_log_with_id[new_id] = self.voltage_log[neuron.uid]
            spike_log_with_id[new_id] = self.spike_log[neuron.uid]
            it += 1

        vis = Visualizer(self.timesteps, spike_log_with_id, voltage_log_with_id, Vt=10, dt=self.dt)
        vis.run()
