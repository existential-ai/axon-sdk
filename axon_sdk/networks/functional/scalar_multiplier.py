from axon_sdk.primitives import SpikingNetworkModule, DataEncoder

from typing import Optional


class ScalarMultiplierNetwork(SpikingNetworkModule):
    """
    Rescales a STICK-coded value by multiplying it by factor:
    output interval = input interval x factor
    If interval x scale is bigger than 1.0, subsequent encoded values are meaningless.

    > IMPORTANT: it's the user's responsability to guarantee that interval x factor is
    smaller than 1.0 for all possible input intervals.
    """

    def __init__(
        self, factor: float, encoder: DataEncoder, module_name: Optional[str] = None
    ):
        """
        The design is based on the Memory Network but with adjusted wacc
        in order to perform the rescaling. Time length fixes with respect
        to the original memory network are also applied: Tsyn missing in
        last->acc2 and recall->output.
        """
        super().__init__(module_name)

        Vt = 10.0
        tm = 100.0
        tf = 20.0
        Tsyn = 1.0
        Tmin = encoder.Tmin

        we = Vt
        wi = -Vt
        wacc = (Vt * tm) / encoder.Tmax
        wacc_long = factor * wacc

        self.input = self.add_neuron(Vt, tm, tf, neuron_name="input")
        self.first = self.add_neuron(Vt, tm, tf, neuron_name="first")
        self.last = self.add_neuron(Vt, tm, tf, neuron_name="last")
        self.acc = self.add_neuron(Vt, tm, tf, neuron_name="acc")
        self.acc2 = self.add_neuron(Vt, tm, tf, neuron_name="acc")
        self.output = self.add_neuron(Vt, tm, tf, neuron_name="output")

        self.connect_neurons(self.input, self.first, "V", we, Tsyn)
        self.connect_neurons(self.input, self.last, "V", 0.5 * we, Tsyn)
        self.connect_neurons(self.first, self.first, "V", wi, Tsyn)
        self.connect_neurons(self.first, self.acc, "ge", wacc_long, Tsyn + Tmin)
        self.connect_neurons(
            self.last, self.acc2, "ge", wacc_long, 2 * Tsyn
        )  # missing Tsyn in the original memory net in STICK paper
        self.connect_neurons(self.acc, self.acc2, "ge", -wacc_long, Tsyn)

        self.connect_neurons(self.acc2, self.output, "V", we, Tsyn + Tmin)
        self.connect_neurons(
            self.acc, self.acc2, "ge", wacc, 2 * Tsyn
        )  # 'ready' triggers 'recall' directly
        self.connect_neurons(
            self.acc, self.output, "V", we, 3 * Tsyn
        )  # 'ready' triggers 'recall' directly; missing Tsyn in the original memory net in STICK paper


if __name__ == "__main__":
    from axon_sdk import Simulator
    
    factor = 100
    val = 0.0042  ## make sure that val x scale < 1
    assert (
        val * factor < 1
    ), "val * scale must be smaller than 1.0, since that's the max. STICK range"

    enc = DataEncoder(Tcod=150)
    rescaler_net = ScalarMultiplierNetwork(factor=factor, encoder=enc, module_name="rescaler")
    sim = Simulator(rescaler_net, enc)

    sim.apply_input_value(value=val, neuron=rescaler_net.input, t0=0)
    sim.simulate(300)

    output_spikes = sim.spike_log[rescaler_net.output.uid]
    assert len(output_spikes) == 2, "Didn't get 2 output spikes"
    interval = output_spikes[1] - output_spikes[0]

    print(f"Input value: {val}")
    print(f"Scale: {factor}")
    print(f"Retrieved value: {enc.decode_interval(interval):.4f}")
