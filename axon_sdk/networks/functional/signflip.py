from axon_sdk.primitives import SpikingNetworkModule, DataEncoder

from typing import Optional

class SignFlipperNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, module_name: Optional[str] = None):
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
        wacc = Vt * tm / encoder.Tmax
        wacc_bar = Vt * tm / encoder.Tcod
        gmult = Vt * tm / tf

        self.inp_plus = self.add_neuron(Vt=Vt, tf=tf, tm=tm, neuron_name='inp_plus')
        self.inp_minus = self.add_neuron(Vt=Vt, tf=tf, tm=tm, neuron_name='inp_minus')
        self.outp_plus = self.add_neuron(Vt=Vt, tf=tf, tm=tm, neuron_name='outp_plus')
        self.outp_minus = self.add_neuron(Vt=Vt, tf=tf, tm=tm, neuron_name='outp_minus')

        self.connect_neurons(self.inp_plus, self.outp_minus, "V", we, Tsyn)
        self.connect_neurons(self.inp_minus, self.outp_plus, "V", we, Tsyn)


if __name__ == "__main__":
    from axon_sdk.simulator import Simulator

    enc = DataEncoder()
    net = SignFlipperNetwork(encoder=enc, module_name='sign_flip_net')
    sim = Simulator(net, enc, dt=0.001)

    value = +0.5
    print(f"Input value: {value}")

    if value >= 0:
        sim.apply_input_value(abs(value), net.inp_plus)
    else:
        sim.apply_input_value(abs(value), net.inp_minus)

    sim.simulate(300)

    spikes_plus = sim.spike_log.get(net.outp_plus.uid, [])
    spikes_minus = sim.spike_log.get(net.outp_minus.uid, [])

    if len(spikes_plus) == 2:
        print(f"Got {len(spikes_plus)} spikes in output plus")
        assert len(spikes_plus) == 2, "Didn't get 2 spikes in output plus!!"
        interval = spikes_plus[1] - spikes_plus[0]
        print(f"Decoded value: {enc.decode_interval(interval)}")

    if len(spikes_minus) == 2:
        print(f"Got {len(spikes_minus)} spikes in output minus")
        assert len(spikes_minus) == 2, "Didn't get 2 spikes in output minus!!"
        interval = spikes_minus[1] - spikes_minus[0]
        print(f"Decoded value: -{enc.decode_interval(interval)}")