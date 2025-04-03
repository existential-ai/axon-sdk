from primitives import SpikingNetworkModule, DataEncoder, ExplicitNeuron
from typing import Optional

class InvertingMemoryNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, name: Optional[str] = None) -> None:
        super().__init__(name)

        Vt = 10.0
        tm = 100.0
        tf = 20.0

        we = Vt
        wi = -Vt
        wacc = Vt * tm / encoder.Tmax
        Tsyn = 1.0
        Tmin = encoder.Tmin
        Tneu = 0.01

        inp = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name='input')
        first = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name='first')
        last = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name='last')
        acc = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name='acc')
        recall = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name='recall')
        output = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name='output')

        self.connect_neurons(inp, first, 'V', we, Tsyn)
        self.connect_neurons(inp, last, 'V', 0.5 * we, Tsyn)
        self.connect_neurons(first, first, 'V', wi, Tsyn)
        self.connect_neurons(first, acc, 'ge', wacc, Tsyn + Tmin)
        self.connect_neurons(last, acc, 'ge', -wacc, Tsyn)
        self.connect_neurons(recall, acc, 'ge', wacc, Tsyn)
        self.connect_neurons(acc, output, 'V', we, Tsyn)
        self.connect_neurons(recall, output, 'V', we, 2 * Tsyn + Tneu)


        self.inp = inp
        self.output = output
        self.recall = recall


from simulator import Simulator

if __name__ == "__main__":
    val = 0.6
    encoder = DataEncoder()
    imn = InvertingMemoryNetwork(encoder)
    sim = Simulator(imn, encoder)
    sim.apply_input_value(value=val, neuron=imn.inp, t0=0)
    sim.apply_input_spike(imn.recall, t=200)
    sim.simulate(simulation_time=300)

    output_spikes = sim.spike_log[imn.output.uid]
    out_val = encoder.decode_interval(output_spikes[1] - output_spikes[0])
    print(f'Input val: {val}')
    print(f'Inverted val (1-val): {out_val}')