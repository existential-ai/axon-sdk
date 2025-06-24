from axon_sdk.primitives import (
    SpikingNetworkModule,
    DataEncoder,
)
from typing import Optional

# Defining constant value here
T_F = 50.0


class LogNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, module_name: Optional[str] = None):
        super().__init__(module_name)
        self.encoder = encoder

        # Constants
        Vt = 10.0
        tm = 100.0
        self.tf = 20.0  # Increasing this makes Log more accurate
        Tsyn = 1.0
        Tmin = encoder.Tmin

        we = Vt
        wi = -Vt
        gmult = (Vt * tm) / self.tf
        wacc_bar = (Vt * tm) / encoder.Tcod

        # Neurons
        self.input = self.add_neuron(Vt=Vt, tm=tm, tf=self.tf, neuron_name="input")
        self.first = self.add_neuron(Vt=Vt, tm=tm, tf=self.tf, neuron_name="first")
        self.last = self.add_neuron(Vt=Vt, tm=tm, tf=self.tf, neuron_name="last")
        self.acc = self.add_neuron(Vt=Vt, tm=tm, tf=self.tf, neuron_name="acc")
        self.output = self.add_neuron(Vt=Vt, tm=tm, tf=self.tf, neuron_name="output")

        # Input triggers
        self.connect_neurons(self.input, self.first, "V", we, Tsyn)
        self.connect_neurons(self.input, self.last, "V", 0.5 * we, Tsyn)

        # Self-inhibit first after spike
        self.connect_neurons(self.first, self.first, "V", wi, Tsyn)

        # Encoding into acc from first and last
        self.connect_neurons(self.first, self.acc, "ge", wacc_bar, Tsyn + Tmin)
        self.connect_neurons(self.last, self.acc, "gate", 1.0, Tsyn)
        self.connect_neurons(self.last, self.acc, "ge", -wacc_bar, Tsyn)
        self.connect_neurons(self.last, self.acc, "gf", gmult, Tsyn)

        # Output driven by acc spike
        self.connect_neurons(self.acc, self.output, "V", we, Tsyn + Tmin)
        self.connect_neurons(self.last, self.output, "V", we, 2 * Tsyn)


def expected_log_output_delay(x, encoder: DataEncoder, tf: float):
    import math

    Tin = encoder.Tmin + x * encoder.Tcod
    try:
        delay = tf * math.log(encoder.Tcod / (Tin - encoder.Tmin))
        Tout = encoder.Tmin + delay
        return Tout
    except ValueError:
        return float("nan")


def decode_logarithm(output_interval, encoder: DataEncoder, tf: float):
    return (encoder.Tmin - output_interval) / tf


if __name__ == "__main__":
    import math
    from axon_sdk import Simulator

    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    lognet = LogNetwork(encoder, module_name="lognet")

    val = 0.389
    sim = Simulator(lognet, encoder, dt=0.01)
    sim.apply_input_value(val, neuron=lognet.input, t0=10)
    sim.simulate(300)

    output_spikes = sim.spike_log.get(lognet.output.uid, [])
    acc_spikes = sim.spike_log.get(lognet.acc.uid, [])
    last_spike_time = sim.spike_log.get(lognet.last.uid, [None])[-1]

    print(f"Input value: {val}, log result: {math.log(val)}")
    print(f"Expected delay (log({val})): {expected_log_output_delay(val, encoder, lognet.tf):.3f} ms")

    if len(output_spikes) >= 2:
        interval = output_spikes[1] - output_spikes[0]
        actual_value = decode_logarithm(interval, encoder, lognet.tf)
        print(
            f"✅ Output spike interval: {interval:.3f} ms, corresponding to {actual_value}"
        )
    elif len(output_spikes) == 1:
        delay = output_spikes[0] - last_spike_time if last_spike_time else float("nan")
        print(
            f"✅ Output spike at: {output_spikes[0]:.3f} ms (delay from 'last': {delay:.3f} ms)"
        )
    else:
        print("❌ No output spike detected.")
