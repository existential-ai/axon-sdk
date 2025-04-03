from primitives import SpikingNetworkModule, DataEncoder, ExplicitNeuron

class MemoryNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder) -> None:
        super().__init__()

        Vt = 10.0
        tm = 100.0
        tf = 20.0

        we = Vt
        wi = -Vt
        Tsyn = 1.0
        Tneu = 0.01
        Tmin = encoder.Tmin
        wacc = Vt * tm / encoder.Tmax

        # Create Neurons
        input = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name='input')
        first = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name='first')
        last = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name='last')
        acc = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name='acc')
        acc2 = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name='acc2')
        recall = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name='recall')
        ready = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name='ready')
        output = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name='output')

        # Connections from input
        self.connect_neurons(input, first, 'V', we, Tsyn)
        self.connect_neurons(input, last, 'V', 0.5 * we, Tsyn)

        # Inhibit first neuron after it spikes
        self.connect_neurons(first, first, 'V', wi, Tsyn)

        # First → acc
        self.connect_neurons(first, acc, 'ge', wacc, Tsyn)

        # Last → acc2 (negative to delay output)
        self.connect_neurons(last, acc2, 'ge', wacc, Tsyn)

        # acc → acc2
        self.connect_neurons(acc, acc2, 'ge', -wacc, Tsyn)

        # Recall → acc2
        self.connect_neurons(recall, acc2, 'ge', wacc, Tsyn)

        # Recall → output
        self.connect_neurons(recall, output, 'V', we, Tsyn)

        # acc2 → output
        self.connect_neurons(acc2, output, 'V', we, Tsyn)

        # Ready → acc2
        self.connect_neurons(ready, acc2, 'V', we, Tsyn)

        # External references
        self.input = input
        self.output = output
        self.recall = recall
        self.ready = ready


if __name__ == '__main__':
    from simulator import Simulator
    val = 1  # test input value
    encoder = DataEncoder(Tcod=100)
    memnet = MemoryNetwork(encoder)

    # Set up simulator
    sim = Simulator(net=memnet, encoder=encoder, dt=0.01)

    # Apply encoded input to 'input' neuron at t=0
    sim.apply_input_value(value=val, neuron=memnet.input, t0=0)

    # Apply recall spike at t=200ms
    sim.apply_input_spike(neuron=memnet.recall, t=0)


    # Run simulation for enough time to capture output
    sim.simulate(simulation_time=200)

    # Retrieve and decode output
    output_spikes = sim.spike_log.get(memnet.output.uid, [])
    if len(output_spikes) >= 2:
        out_val = encoder.decode_interval(output_spikes[1] - output_spikes[0])
        print(f'✅ Input value: {val:.3f}')
        print(f'✅ Recalled value: {out_val:.3f}')
        print(f'✅ Interval: {output_spikes[1] - output_spikes[0]:.3f} ms')
    else:
        print(f'❌ Output spike missing or incomplete: {output_spikes}')