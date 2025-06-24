from axon_sdk.networks import SubtractorNetwork, ExponentialNetwork
from axon_sdk.primitives import DataEncoder, SpikingNetworkModule

from typing import Optional


class LogAndSyncNetwork(SpikingNetworkModule):
    """
    > IMPORTANT: Do not use this network outside of this file since it's tailor made for the div module.

    Inspired by the log network and multiplier network (discussed in the STICK paper),
    it computes the log of both inputs and outputs them synced in time.

    For each of its inputs (x1 and x2, in [0,1]), the output is a pair of spikes with a time interval of
    Tmin + tf * (-ln(xi))

    (Note ln(xi) < 0)
    """

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

        # First input
        self.input1 = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name="input1")
        self.first1 = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name="first1")
        self.last1 = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name="last1")
        self.acc1 = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name="acc1")
        self.output1 = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name="output1")

        self.connect_neurons(self.input1, self.first1, "V", we, Tsyn)
        self.connect_neurons(self.first1, self.first1, "V", wi, Tsyn)
        self.connect_neurons(self.input1, self.last1, "V", 0.5 * we, Tsyn)
        self.connect_neurons(self.first1, self.acc1, "ge", wacc_bar, Tsyn + Tmin)
        self.connect_neurons(self.last1, self.acc1, "ge", -wacc_bar, Tsyn)

        # Second input
        self.input2 = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name="input2")
        self.first2 = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name="first2")
        self.last2 = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name="last2")
        self.acc2 = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name="acc2")
        self.output2 = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name="output2")

        self.connect_neurons(self.input2, self.first2, "V", we, Tsyn)
        self.connect_neurons(self.first2, self.first2, "V", wi, Tsyn)
        self.connect_neurons(self.input2, self.last2, "V", 0.5 * we, Tsyn)

        self.connect_neurons(self.first2, self.acc2, "ge", wacc_bar, Tsyn + Tmin)
        self.connect_neurons(self.last2, self.acc2, "ge", -wacc_bar, Tsyn)

        # Sync neuron
        self.sync = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name="sync")
        # sync neuron isused like in the multiplier module. A Synchronizer module cannot
        # be used since the intervals could be larger than Tmax)
        self.connect_neurons(self.last1, self.sync, "V", 0.5 * we, Tsyn)
        self.connect_neurons(self.last2, self.sync, "V", 0.5 * we, Tsyn)

        self.connect_neurons(self.sync, self.output1, "V", we, 2 * Tsyn)
        self.connect_neurons(self.sync, self.output2, "V", we, 2 * Tsyn)

        self.connect_neurons(self.sync, self.acc1, "gf", gmult, Tsyn)
        self.connect_neurons(self.sync, self.acc1, "gate", 1, Tsyn)
        self.connect_neurons(self.acc1, self.output1, "V", we, Tsyn + Tmin)

        self.connect_neurons(self.sync, self.acc2, "gf", gmult, Tsyn)
        self.connect_neurons(self.sync, self.acc2, "gate", 1, Tsyn)
        self.connect_neurons(self.acc2, self.output2, "V", we, Tsyn + Tmin)


class DivNetwork(SpikingNetworkModule):
    """
    DivNetworkModule:
    Given inputs x1 and x2 between [0, 1] such that x1 <= x2, it computes x1 / x2.
    Experiments show 4 decimal places of precision.

    > IMPORTANT: It's the user responsability to guarantee that x1 <= x2.
    Otherwise, The network will malfunction (it will not output any spikes)

    Its working principle is based on the identity x1/x2 = exp(ln(x1) - ln(x2)).

    It uses two log modules whose outputs are synced, a sub network to generate the difference
    and a exp network to compute back the exponent.
    """

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

        logsync = LogAndSyncNetwork(encoder=encoder, module_name="log_and_sync_net")
        sub = SubtractorNetwork(encoder=encoder, module_name="sub_net")
        exp = ExponentialNetwork(encoder=encoder, module_name="exp_net")

        self.add_subnetwork(logsync)
        self.add_subnetwork(sub)
        self.add_subnetwork(exp)

        self.input1 = logsync.input1
        self.input2 = logsync.input2
        self.output = exp.output

        self.connect_neurons(logsync.output1, sub.input1, "V", we, Tsyn)
        self.connect_neurons(logsync.output2, sub.input2, "V", we, Tsyn)

        # Since x1 < x2 (by construction), sub always spikes on output plus
        self.connect_neurons(sub.output_plus, exp.input, "V", we, Tsyn)


if __name__ == "__main__":
    from axon_sdk import Simulator

    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    net = DivNetwork(encoder, module_name="div")
    sim = Simulator(net, encoder, dt=0.001)

    x1 = 0.0001
    x2 = 0.5

    assert x1 < x2, f"x1 must be smaller than x2!!, but x1: {x1} and x2 {x2}"

    sim.apply_input_value(value=x1, neuron=net.input1, t0=0)
    sim.apply_input_value(value=x2, neuron=net.input2, t0=0)
    sim.simulate(simulation_time=300)

    # --- Output Decoding ---
    out_spikes = sim.spike_log[net.output.uid]
    assert len(out_spikes) == 2, "Didn't get 2 output spikes"

    print(f"Input x1: {x1}, input x2: {x2}")
    print(f"Expected {(x1 / x2):.4f}")
    print(f"Decoded: {encoder.decode_interval(out_spikes[1] - out_spikes[0])}")
