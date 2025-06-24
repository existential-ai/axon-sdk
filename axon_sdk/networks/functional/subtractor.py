from axon_sdk.primitives import (
    SpikingNetworkModule,
    DataEncoder,
)


class SubtractorNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, module_name=None):
        super().__init__(module_name)
        self.encoder = encoder

        # Parameters
        Vt = 10.0
        tm = 100.0
        tf = 20.0
        Tsyn = 1.0
        Tneu = 0.01
        Tmin = encoder.Tmin

        we = Vt
        wi = -Vt

        # Create all neurons
        self.input1 = self.add_neuron(Vt, tm, tf, neuron_name="input1")
        self.input2 = self.add_neuron(Vt, tm, tf, neuron_name="input2")

        self.sync1 = self.add_neuron(Vt, tm, tf, neuron_name="sync1")
        self.sync2 = self.add_neuron(Vt, tm, tf, neuron_name="sync2")

        self.inb1 = self.add_neuron(Vt, tm, tf, neuron_name="inb1")
        self.inb2 = self.add_neuron(Vt, tm, tf, neuron_name="inb2")

        self.output_plus = self.add_neuron(Vt, tm, tf, neuron_name="output_plus")
        self.output_minus = self.add_neuron(Vt, tm, tf, neuron_name="output_minus")

        self.zero = self.add_neuron(Vt, tm, tf, neuron_name="zero")

        # --- Main black synapse pathway ---
        # Inputs to syncs
        self.connect_neurons(self.input1, self.sync1, "V", 0.5 * we, Tsyn)
        self.connect_neurons(self.input2, self.sync2, "V", 0.5 * we, Tsyn)

        # sync → inb
        self.connect_neurons(self.sync1, self.inb1, "V", we, Tsyn)
        self.connect_neurons(self.sync2, self.inb2, "V", we, Tsyn)

        self.connect_neurons(self.sync1, self.inb2, "V", wi, Tsyn)
        self.connect_neurons(self.sync2, self.inb1, "V", wi, Tsyn)

        # sync1/2 → output+/- direct
        self.connect_neurons(
            self.sync1, self.output_plus, "V", we, Tmin + (3 * Tsyn + 2 * Tneu)
        )
        self.connect_neurons(
            self.sync2, self.output_minus, "V", we, Tmin + (3 * Tsyn + 2 * Tneu)
        )

        self.connect_neurons(self.sync2, self.output_plus, "V", we, 3 * Tsyn + 2 * Tneu)
        self.connect_neurons(
            self.sync1, self.output_minus, "V", we, 3 * Tsyn + 2 * Tneu
        )

        # inb1 → output-
        self.connect_neurons(self.inb2, self.output_minus, "V", 2 * wi, Tsyn)

        # inb2 → output+
        self.connect_neurons(self.inb1, self.output_plus, "V", 2 * wi, Tsyn)

        # inb1 → output+ (excitatory leak)
        self.connect_neurons(self.output_plus, self.inb2, "V", 0.5 * we, Tsyn)
        self.connect_neurons(self.output_minus, self.inb1, "V", 0.5 * we, Tsyn)

        # zero self excitation and inhibition
        self.connect_neurons(self.zero, self.zero, "V", we, Tneu)

        # zero to sync1, sync2
        self.connect_neurons(self.sync1, self.zero, "V", 0.5 * wi, 2 * Tneu)
        self.connect_neurons(self.sync2, self.zero, "V", 0.5 * wi, 2 * Tneu)

        self.connect_neurons(self.sync1, self.zero, "V", 0.5 * we, Tneu)
        self.connect_neurons(self.sync2, self.zero, "V", 0.5 * we, Tneu)

        # zero to output+
        self.connect_neurons(self.zero, self.output_minus, "V", 2 * wi, Tneu)
        self.connect_neurons(self.zero, self.inb2, "V", wi, Tneu)

        # Save outputs
        self.output = self.output_plus
        self.output_alt = self.output_minus


if __name__ == "__main__":
    from axon_sdk.simulator import Simulator

    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = SubtractorNetwork(encoder, module_name='sub')
    sim = Simulator(net, encoder, dt=0.01)

    x1 = 0.73
    x2 = 0.62

    sim.apply_input_value(value=x1, neuron=net.input1, t0=0)
    sim.apply_input_value(value=x2, neuron=net.input2, t0=0)

    sim.simulate(simulation_time=300)

    # --- Output Decoding ---
    spikes_plus = sim.spike_log.get(net.output.uid, [])
    spikes_minus = sim.spike_log.get(net.output_alt.uid, [])

    print(f"Input1  : {x1}")
    print(f"Input2  : {x2}")
    print(f"Expected: {x1 - x2}")

    if len(spikes_plus) >= 2:
        interval = spikes_plus[1] - spikes_plus[0]
        decoded = encoder.decode_interval(interval)
        print(f"Output: x1 - x2 = {decoded}")
        print(f"(Neuron 'output+' spiked)")
    elif len(spikes_minus) >= 2:
        interval = spikes_minus[1] - spikes_minus[0]
        decoded = encoder.decode_interval(interval)
        print(f"Output: x1 - x2 = -{decoded}")
        print(f"(Neuron 'output-' spiked)")
    else:
        print("No valid output spikes detected")
