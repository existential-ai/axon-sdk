from axon_sdk.primitives import (
    SpikingNetworkModule,
    DataEncoder,
)
from axon_sdk.networks import MemoryNetwork

from typing import Optional


class SynchronizerNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, N: int, module_name: Optional[str] = None):
        super().__init__(module_name)
        self.encoder = encoder

        Vt = 10.0
        tm = 100.0
        tf = 20.0

        we = Vt
        Tsyn = 1.0

        # Create sync neuron
        self.sync = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="sync"
        )

        self.input_neurons = []
        self.output_neurons = []
        self.memory_blocks = []

        for i in range(N):
            # Input and output interface neurons
            input_neuron = self.add_neuron(
                Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name=f"input_{i}"
            )
            output_neuron = self.add_neuron(
                Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name=f"output_{i}"
            )
            self.input_neurons.append(input_neuron)
            self.output_neurons.append(output_neuron)

            # Memory network block
            memory = MemoryNetwork(encoder, module_name=f"mem_{i}")
            self.add_subnetwork(memory)
            self.memory_blocks.append(memory)

            # Connect input to memory.input
            self.connect_neurons(input_neuron, memory.input, "V", we, Tsyn)

            # Connect memory.output to output neuron
            self.connect_neurons(memory.output, output_neuron, "V", we, Tsyn)

            # Connect memory.ready to sync
            self.connect_neurons(
                memory.ready, self.sync, "V", (we / N) + 0.0001, Tsyn
            )

            # Connect sync to memory.recall
            self.connect_neurons(self.sync, memory.recall, "V", we, Tsyn)


if __name__ == "__main__":
    from axon_sdk.simulator import Simulator
    import random

    encoder = DataEncoder()
    N = 4  # Number of synchronized inputs
    values = [random.random() for i in range(N)]  # N values between 0–1
    t0s = [random.random() * 20 for i in range(N)]  # N different starting points

    syncnet = SynchronizerNetwork(encoder, N=N)
    sim = Simulator(net=syncnet, encoder=encoder, dt=0.01)
    
    for i, val in enumerate(values):
        sim.apply_input_value(value=val, neuron=syncnet.input_neurons[i], t0=t0s[i])

    sim.simulate(simulation_time=300)

    for i, in_neuron in enumerate(syncnet.input_neurons):
        spikes = sim.spike_log.get(in_neuron.uid, [])
        if len(spikes) >= 2:
            interval = spikes[1] - spikes[0]
            decoded = encoder.decode_interval(interval)
            print(
                f"✅ Input[{i}] | Spike1 time: {spikes[0]:.3f} | Decoded value {decoded:.4f}"
            )
        else:
            print(f"❌ Output[{i}] missing second spike: {spikes}")

    print("\n")

    for i, out_neuron in enumerate(syncnet.output_neurons):
        spikes = sim.spike_log.get(out_neuron.uid, [])
        if len(spikes) >= 2:
            interval = spikes[1] - spikes[0]
            decoded = encoder.decode_interval(interval)
            print(
                f"✅ Output[{i}] | Spike1 time: {spikes[0]:.3f} | Decoded value {decoded:.4f}"
            )
        else:
            print(f"❌ Output[{i}] missing second spike: {spikes}")
