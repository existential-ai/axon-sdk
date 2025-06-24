from axon_sdk.primitives import (
    SpikingNetworkModule,
    DataEncoder,
)
from typing import Optional


class ConstantNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, value: float, module_name: Optional[str] = None) -> None:
        super().__init__(module_name)
        self.encoder = encoder
        self.value = value

        Vt = 10.0
        tm = 100.0
        tf = 20.0
        we = Vt
        Tsyn = 1.0
        f_x = (value * self.encoder.Tcod) + encoder.Tmin

        # Create constant neuron
        self.recall = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="recall"
        )
        self.output = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, Vreset=0.0, neuron_name="output"
        )

        # Connect constant neuron to itself with a delay
        self.connect_neurons(self.recall, self.output, "V", we, Tsyn)
        self.connect_neurons(self.recall, self.output, "V", we, Tsyn + f_x)


if __name__ == "__main__":
    from axon_sdk import Simulator

    encoder = DataEncoder()
    value = 0  # Constant value between 0–1

    constant_network = ConstantNetwork(encoder, value)
    sim = Simulator(constant_network, encoder)
    sim.apply_input_spike(constant_network.recall, t=0)
    sim.simulate(simulation_time=100)
    output_spikes = sim.spike_log[constant_network.output.uid]
    print(f"Input value: {value}")
    print(f"Output spikes: {output_spikes}")
