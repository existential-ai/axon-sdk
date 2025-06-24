from axon_sdk.primitives import (
    SpikingNetworkModule,
    DataEncoder,
)
from axon_sdk.networks import MultiplierNetwork, ScalarMultiplierNetwork
from typing import Optional


class SignedMultiplierNormNetwork(SpikingNetworkModule):
    """
    Module composed of a signed multiplier followed by a scalar multiplier.
    
    Given two input values x1 and x2 in the range [0,1] and a factor in the range [0,100],
    the output is:
    
    output = (x1 * x2) * factor

    > IMPORTANT: It's the user's responsability to guarantee that (x1 * x2) * factor < 1.
    Otherwise, the network will misbehave due to the overflow.

    This block is useful to perform multiplication over an extended range of inputs beyond [0,1].
    For example, to multiply 13 and 7 the block can be used as follows:

    x1 = 13 / 100
    x2 = 7 / 100
    factor = 100

    output = (13 * 7) / 100

    Denormalizing the output by multiplying by 100 will yield a result in the original range
    """
    def __init__(self, encoder: DataEncoder, factor: float, module_name: Optional[str] = None) -> None:
        super().__init__(module_name)
        self.encoder = encoder

        # Parameters
        Vt = 10.0
        tm = 100.0
        tf = 20.0
        Tsyn = 1.0
        Tmin = encoder.Tmin

        we = Vt
        wi = -Vt

        # Create multiplier network
        self.mn = MultiplierNetwork(encoder, module_name='mul_net')
        self.add_subnetwork(self.mn)

        self.scmn = ScalarMultiplierNetwork(factor=factor, encoder=encoder, module_name='scalar_mul')
        self.add_subnetwork(self.scmn)

        self.input1_plus = self.add_neuron(
            Vt, tm, tf, neuron_name="input1_plus"
        )
        self.input1_minus = self.add_neuron(
            Vt, tm, tf, neuron_name="input1_minus"
        )
        self.input2_plus = self.add_neuron(
            Vt, tm, tf, neuron_name="input2_plus"
        )
        self.input2_minus = self.add_neuron(
            Vt, tm, tf, neuron_name="input2_minus"
        )

        self.output_plus = self.add_neuron(
            Vt, tm, tf, neuron_name="output_plus"
        )
        self.output_minus = self.add_neuron(
            Vt, tm, tf, neuron_name="output_minus"
        )

        self.sign1 = self.add_neuron(Vt, tm, tf, neuron_name="sign1")
        self.sign2 = self.add_neuron(Vt, tm, tf, neuron_name="sign2")
        self.sign3 = self.add_neuron(Vt, tm, tf, neuron_name="sign3")
        self.sign4 = self.add_neuron(Vt, tm, tf, neuron_name="sign4")

        self.connect_neurons(self.input1_plus, self.mn.input1, "V", we, Tsyn)
        self.connect_neurons(self.input1_minus, self.mn.input1, "V", we, Tsyn)

        self.connect_neurons(self.input2_plus, self.mn.input2, "V", we, Tsyn)
        self.connect_neurons(self.input2_minus, self.mn.input2, "V", we, Tsyn)

        self.connect_neurons(self.input1_plus, self.sign1, "V", 0.25 * we, Tsyn)
        self.connect_neurons(self.input1_plus, self.sign3, "V", 0.25 * we, Tsyn)

        self.connect_neurons(self.input1_minus, self.sign2, "V", 0.25 * we, Tsyn)
        self.connect_neurons(self.input1_minus, self.sign4, "V", 0.25 * we, Tsyn)

        self.connect_neurons(self.input2_plus, self.sign1, "V", 0.25 * we, Tsyn)
        self.connect_neurons(self.input2_plus, self.sign2, "V", 0.25 * we, Tsyn)

        self.connect_neurons(self.input2_minus, self.sign3, "V", 0.25 * we, Tsyn)
        self.connect_neurons(self.input2_minus, self.sign4, "V", 0.25 * we, Tsyn)

        self.connect_neurons(self.sign1, self.sign2, "V", 0.5 * wi, Tsyn)
        self.connect_neurons(self.sign1, self.sign3, "V", 0.5 * wi, Tsyn)

        self.connect_neurons(self.sign2, self.sign1, "V", 0.5 * wi, Tsyn)
        self.connect_neurons(self.sign2, self.sign4, "V", 0.5 * wi, Tsyn)

        self.connect_neurons(self.sign3, self.sign1, "V", 0.5 * wi, Tsyn)
        self.connect_neurons(self.sign3, self.sign4, "V", 0.5 * wi, Tsyn)

        self.connect_neurons(self.sign4, self.sign2, "V", 0.5 * wi, Tsyn)
        self.connect_neurons(self.sign4, self.sign3, "V", 0.5 * wi, Tsyn)

        self.connect_neurons(self.mn.output, self.scmn.input, "V", we, Tsyn)
        self.connect_neurons(self.scmn.output, self.output_plus, "V", we, Tsyn)
        self.connect_neurons(self.scmn.output, self.output_minus, "V", we, Tsyn)

        self.connect_neurons(self.sign1, self.output_minus, "V", 2 * wi, Tsyn)
        self.connect_neurons(self.sign2, self.output_plus, "V", 2 * wi, Tsyn)
        self.connect_neurons(self.sign3, self.output_plus, "V", 2 * wi, Tsyn)
        self.connect_neurons(self.sign4, self.output_minus, "V", 2 * wi, Tsyn)


if __name__ == "__main__":
    from axon_sdk.simulator import Simulator

    norm = 100

    encoder = DataEncoder(Tmin=10.0, Tcod=150.0)
    net = SignedMultiplierNormNetwork(encoder, factor=norm, module_name='signmul_net')
    sim = Simulator(net, encoder, dt=0.0005)

    x1 = 42.32
    x2 = 2.15
    
    true_product = x1 * x2

    val1 = x1 / norm
    val2 = x2 / norm

    # Apply both input values
    if val1 > 0:
        sim.apply_input_value(abs(val1), neuron=net.input1_plus, t0=0)
    else:
        sim.apply_input_value(abs(val1), neuron=net.input1_minus, t0=0)

    if val2 > 0:
        sim.apply_input_value(abs(val2), neuron=net.input2_plus, t0=0)
    else:
        sim.apply_input_value(abs(val2), neuron=net.input2_minus, t0=0)

    # Simulate long enough to see output
    sim.simulate(simulation_time=500)

    spikes_plus = sim.spike_log.get(net.output_plus.uid, [])
    spikes_minus = sim.spike_log.get(net.output_minus.uid, [])

    if len(spikes_plus) == 2:
        print("Output + has 2 spikes")
        interval = spikes_plus[1] - spikes_plus[0]
        decoded_val = encoder.decode_interval(interval)
    if len(spikes_minus) == 2:
        print("Output - has 2 spikes")
        interval = spikes_minus[1] - spikes_minus[0]
        decoded_val = -1 * encoder.decode_interval(interval)

    if len(spikes_plus) != 2 and len(spikes_minus) != 2:
        print(f"❌ Output spike missing or incomplete")
        raise ValueError


    print(f"✅ Input: {x1} × {x2}")
    print(f"✅ Expected: {true_product:.4f}")
    print(f"✅ Output spike interval: {interval:.3f} ms")
    print(f"✅ Decoded value: {decoded_val:.4f}")
    print(f"✅ Decoded and Denormalized value : {norm * decoded_val:.4f}")

    print("multiplier decoded val")
    spikes = sim.spike_log.get(net.mn.output.uid, [])
    print(encoder.decode_interval(spikes[1] - spikes[0]))

        

