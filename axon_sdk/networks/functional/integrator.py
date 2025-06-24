from axon_sdk.primitives import (
    SpikingNetworkModule,
    DataEncoder,
)
from axon_sdk.networks import (
    SignedConstantNetwork,
    LinearCombinatorNetwork,
)


class IntegratorNetwork(SpikingNetworkModule):
    def __init__(
        self,
        encoder: DataEncoder,
        constant: float,
        coeffs: list,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.encoder = encoder

        # Constants
        Vt = 10.0
        tm = 100.0
        tf = 20.0  ## Increasing this makes Log more accurate
        Tsyn = 1.0
        Tmin = encoder.Tmin

        we = Vt
        wi = -Vt
        gmult = (Vt * tm) / tf
        wacc = (Vt * tm) / encoder.Tcod

        self.init = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name="init")

        self.input_plus = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "input_plus"
        )
        self.input_minus = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "input_minus"
        )
        self.start = self.add_neuron(Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "start")
        self.output_plus = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "output_plus"
        )
        self.output_minus = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "output_minus"
        )
        self.new_input = self.add_neuron(
            Vt=Vt, tm=tm, tf=tf, neuron_name=prefix + "new_input"
        )

        self.constant_network = SignedConstantNetwork(encoder, constant, prefix + "sc")
        self.lin_comb = LinearCombinatorNetwork(
            encoder, 2, coeffs, prefix=prefix + "lin_comb"
        )

        self.add_subnetwork(self.constant_network)
        self.add_subnetwork(self.lin_comb)

        self.connect_neurons(self.init, self.constant_network.recall, "V", we, Tsyn)

        self.connect_neurons(
            self.input_plus, self.lin_comb.input_neurons[2], "V", we, Tsyn
        )
        self.connect_neurons(
            self.input_minus, self.lin_comb.input_neurons[3], "V", we, Tsyn
        )

        self.connect_neurons(self.start, self.lin_comb.input_neurons[2], "V", we, Tsyn)
        self.connect_neurons(
            self.start, self.lin_comb.input_neurons[2], "V", we, Tsyn + Tmin
        )

        self.connect_neurons(
            self.constant_network.output_plus,
            self.lin_comb.input_neurons[0],
            "V",
            we,
            Tsyn,
        )
        self.connect_neurons(
            self.constant_network.output_minus,
            self.lin_comb.input_neurons[1],
            "V",
            we,
            Tsyn,
        )

        self.connect_neurons(
            self.lin_comb.output_plus,
            self.lin_comb.input_neurons[0],
            "V",
            we,
            Tsyn,
        )
        self.connect_neurons(
            self.lin_comb.output_minus,
            self.lin_comb.input_neurons[1],
            "V",
            we,
            Tsyn,
        )

        self.connect_neurons(self.lin_comb.output_plus, self.output_plus, "V", we, Tsyn)
        self.connect_neurons(
            self.lin_comb.output_minus, self.output_minus, "V", we, Tsyn
        )

        self.connect_neurons(self.lin_comb.start, self.new_input, "V", we, Tsyn)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from axon_sdk import Simulator

    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    x0 = 0.3
    dt = 0.2
    coeffs = [1.0, dt]  # Integrate: x[t+1] = x[t] + dt * u[t]
    net = IntegratorNetwork(encoder, constant=x0, coeffs=coeffs)

    sim = Simulator(net, encoder, dt=0.01)

    # Step 1: Trigger init to recall x0
    sim.apply_input_spike(net.init, t=0)

    # Step 2: Apply first external input (u[0] = 0.5) at t=50
    sim.apply_input_value(0.5, net.input_plus, t0=50)

    # Step 3: Step again (e.g., t=200)
    sim.apply_input_spike(net.start, t=200)
    sim.apply_input_value(0.1, net.input_plus, t0=200)

    # Step 4: Step again (t=400)
    sim.apply_input_spike(net.start, t=400)
    sim.apply_input_value(0.2, net.input_plus, t0=400)

    # Run simulation
    sim.simulate(800)

    # Decode final output
    output_plus = sim.spike_log.get(net.output_plus.uid, [])
    output_minus = sim.spike_log.get(net.output_minus.uid, [])

    if len(output_plus) >= 2:
        interval = output_plus[1] - output_plus[0]
        value = encoder.decode_interval(interval)
        print(f"✅ Integrated Value (+): {value:.3f}")
    elif len(output_minus) >= 2:
        interval = output_minus[1] - output_minus[0]
        value = -encoder.decode_interval(interval)
        print(f"✅ Integrated Value (−): {value:.3f}")
    else:
        print("❌ No output interval detected.")

    # Optionally print spike log
    print("\n📊 Spike log summary:")
    for k, v in sim.spike_log.items():
        print(f"{k}: {v}")
