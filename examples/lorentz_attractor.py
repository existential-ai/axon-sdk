from axon_sdk.primitives import SpikingNetworkModule, ExplicitNeuron, DataEncoder
from axon_sdk.networks.functional.signed_multiplier import SignedMultiplierNetwork
from axon_sdk.networks.functional.integrator import IntegratorNetwork
from axon_sdk.networks.functional.linear_combinator import LinearCombinatorNetwork
from axon_sdk.networks.connecting.synchronizer import SynchronizerNetwork

class LorentzAttractor(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, sigma: float, rho: float, beta: float, prefix: str = '') -> None:
        super().__init__()
        self.encoder = encoder

        # Parameters
        Vt = 10.0
        tm = 100.0
        tf = 20.0
        Tsyn = 1.0
        Tmin = encoder.Tmin
        dt_const = 0.1

        we = Vt

        # Create signed multiplier network
        self.sm_1 = SignedMultiplierNetwork(encoder, prefix=prefix+ 'mult_1')
        self.sm_2 = SignedMultiplierNetwork(encoder, prefix=prefix + 'mult_2')
        self.add_subnetwork(self.sm_1)
        self.add_subnetwork(self.sm_2)

        # Create integrator network for x, y, and z components
        self.integrator_x = IntegratorNetwork(encoder, constant=dt_const, coeffs=[1.0, 1.0], prefix=prefix + 'intx_')
        self.integrator_y = IntegratorNetwork(encoder, constant=dt_const, coeffs=[1.0, 1.0], prefix=prefix + 'inty_')
        self.integrator_z = IntegratorNetwork(encoder, constant=dt_const, coeffs=[1.0, 1.0], prefix=prefix + 'intz_')

        self.add_subnetwork(self.integrator_x)
        self.add_subnetwork(self.integrator_y)
        self.add_subnetwork(self.integrator_z)

        self.lin_comb_sigma = LinearCombinationNetwork(encoder, 2, [sigma, -sigma], prefix=prefix + 'lcro_')
        self.lin_comb_beta = LinearCombinationNetwork(encoder, 2, [1.0, beta], prefix=prefix + 'lcbeta_')
        self.lin_comb_rho = LinearCombinationNetwork(encoder, 3, [rho, -1.0, -1.0], prefix=prefix + 'lcp_')

        self.add_subnetwork(self.lin_comb_rho)
        self.add_subnetwork(self.lin_comb_beta)
        self.add_subnetwork(self.lin_comb_sigma)

        # Create synchronizer network
        self.sync_network = SynchronizerNetwork(encoder, 3)
        
        self.add_subnetwork(self.sync_network)

        self.output_x_plus = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id=prefix + "output_x_plus")
        self.output_y_plus = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id=prefix + "output_y_plus")
        self.output_z_plus = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id=prefix + "output_z_plus")

        self.output_x_minus = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id=prefix + "output_x_minus")
        self.output_y_minus = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id=prefix + "output_y_minus")
        self.output_z_minus = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id=prefix + "output_z_minus")


        self.global_start = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id=prefix + "start")
        self.global_init = ExplicitNeuron(Vt=Vt, tm=tm, tf=tf, neuron_id=prefix + "init")

        self.add_neurons([
            self.output_x_plus, self.output_y_plus, self.output_z_plus,
            self.output_x_minus, self.output_y_minus, self.output_z_minus,
            self.global_start, self.global_init
        ])

        self.connect_neurons(self.sm_1.output_plus, self.lin_comb_rho.input_neurons[4], 'V', we, Tsyn)
        self.connect_neurons(self.sm_1.output_minus, self.lin_comb_rho.input_neurons[5], 'V', we, Tsyn)

        self.connect_neurons(self.sm_2.output_plus, self.lin_comb_beta.input_neurons[0], 'V', we, Tsyn)
        self.connect_neurons(self.sm_2.output_minus, self.lin_comb_beta.input_neurons[1], 'V', we, Tsyn)

        self.connect_neurons(self.lin_comb_beta.output_plus, self.sync_network.input_neurons[0], 'V', we, Tsyn)
        self.connect_neurons(self.lin_comb_beta.output_minus, self.sync_network.input_neurons[0], 'V', we, Tsyn)

        self.connect_neurons(self.lin_comb_rho.output_plus, self.sync_network.input_neurons[1], 'V', we, Tsyn)
        self.connect_neurons(self.lin_comb_rho.output_minus, self.sync_network.input_neurons[1], 'V', we, Tsyn)

        self.connect_neurons(self.lin_comb_beta.output_plus, self.sync_network.input_neurons[2], 'V', we, Tsyn)
        self.connect_neurons(self.lin_comb_beta.output_minus, self.sync_network.input_neurons[2], 'V', we, Tsyn)

        self.connect_neurons(self.sync_network.output_neurons[2], self.integrator_z.input_plus, 'V', we, Tsyn)
        self.connect_neurons(self.sync_network.output_neurons[2], self.integrator_z.input_minus, 'V', we, Tsyn)

        self.connect_neurons(self.sync_network.output_neurons[1], self.integrator_y.input_plus, 'V', we, Tsyn)
        self.connect_neurons(self.sync_network.output_neurons[1], self.integrator_y.input_minus, 'V', we, Tsyn)

        self.connect_neurons(self.sync_network.output_neurons[0], self.integrator_x.input_plus, 'V', we, Tsyn)
        self.connect_neurons(self.sync_network.output_neurons[0], self.integrator_x.input_minus, 'V', we, Tsyn)

        self.connect_neurons(self.integrator_x.output_plus, self.output_x_plus, 'V', we, Tsyn)
        self.connect_neurons(self.integrator_x.output_minus, self.output_x_minus, 'V', we, Tsyn)

        self.connect_neurons(self.integrator_y.output_plus, self.output_y_plus, 'V', we, Tsyn)
        self.connect_neurons(self.integrator_y.output_minus, self.output_y_minus, 'V', we, Tsyn)

        self.connect_neurons(self.integrator_z.output_plus, self.output_z_plus, 'V', we, Tsyn)
        self.connect_neurons(self.integrator_z.output_minus, self.output_z_minus, 'V', we, Tsyn)

        self.connect_neurons(self.integrator_x.output_plus, self.sm_1.input1_plus, 'V', we, Tsyn)
        self.connect_neurons(self.integrator_x.output_minus, self.sm_1.input1_minus, 'V', we, Tsyn)

        self.connect_neurons(self.integrator_x.output_plus, self.sm_2.input1_plus, 'V', we, Tsyn)
        self.connect_neurons(self.integrator_x.output_minus, self.sm_2.input1_minus, 'V', we, Tsyn)

        self.connect_neurons(self.integrator_x.output_plus, self.lin_comb_rho.input_neurons[0], 'V', we, Tsyn)
        self.connect_neurons(self.integrator_x.output_minus, self.lin_comb_rho.input_neurons[1], 'V', we, Tsyn)

        self.connect_neurons(self.integrator_x.output_plus, self.lin_comb_sigma.input_neurons[2], 'V', we, Tsyn)
        self.connect_neurons(self.integrator_x.output_minus, self.lin_comb_sigma.input_neurons[3], 'V', we, Tsyn)

        self.connect_neurons(self.integrator_y.output_plus, self.lin_comb_sigma.input_neurons[0], 'V', we, Tsyn)
        self.connect_neurons(self.integrator_y.output_minus, self.lin_comb_sigma.input_neurons[1], 'V', we, Tsyn)

        self.connect_neurons(self.integrator_y.output_plus, self.lin_comb_rho.input_neurons[2], 'V', we, Tsyn)
        self.connect_neurons(self.integrator_y.output_minus, self.lin_comb_rho.input_neurons[3], 'V', we, Tsyn)

        self.connect_neurons(self.integrator_y.output_plus, self.sm_2.input2_plus, 'V', we, Tsyn)
        self.connect_neurons(self.integrator_y.output_minus, self.sm_2.input2_minus, 'V', we, Tsyn)

        self.connect_neurons(self.integrator_z.output_plus, self.sm_1.input2_plus, 'V', we, Tsyn)
        self.connect_neurons(self.integrator_z.output_minus, self.sm_1.input2_minus, 'V', we, Tsyn)

        self.connect_neurons(self.integrator_z.output_plus, self.lin_comb_beta.input_neurons[2], 'V', we, Tsyn)
        self.connect_neurons(self.integrator_z.output_minus, self.lin_comb_beta.input_neurons[3], 'V', we, Tsyn)

        self.connect_neurons(self.global_init, self.integrator_x.init, 'V', we, Tsyn)
        self.connect_neurons(self.global_init, self.integrator_y.init, 'V', we, Tsyn)
        self.connect_neurons(self.global_init, self.integrator_z.init, 'V', we, Tsyn)

        self.connect_neurons(self.global_start, self.integrator_x.start, 'V', we, Tsyn)
        self.connect_neurons(self.global_start, self.integrator_y.start, 'V', we, Tsyn)
        self.connect_neurons(self.global_start, self.integrator_z.start, 'V', we, Tsyn)

        self.connect_neurons(self.global_start, self.lin_comb_beta.start, 'V', we, Tsyn)
        self.connect_neurons(self.global_start, self.lin_comb_rho.start, 'V', we, Tsyn)
        self.connect_neurons(self.global_start, self.lin_comb_sigma.start, 'V', we, Tsyn)


def decode_interval(spikes, encoder):
    if len(spikes) >= 2:
        interval = spikes[1] - spikes[0]
        return encoder.decode_interval(interval)
    return None

if __name__ == '__main__':
    from simulator import Simulator
    import matplotlib.pyplot as plt

    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    net = LorentzAttractor(encoder, sigma, rho, beta)
    sim = Simulator(net, encoder, dt=0.1)

    sim.apply_input_spike(net.global_init, t=0)

    # Apply periodic start pulses to advance integration
    #for i in range(2):
    #    sim.apply_input_spike(net.global_start, t=500 + i * 2000)

    sim.apply_input_spike(net.integrator_x.start, t=100)
    sim.apply_input_spike(net.integrator_y.start, t=100)
    sim.apply_input_spike(net.integrator_z.start, t=100)
    sim.apply_input_spike(net.lin_comb_sigma.start, t=100)
    sim.apply_input_spike(net.lin_comb_rho.start, t=100)
    sim.apply_input_spike(net.lin_comb_beta.start, t=100)
    sim.simulate(2000)

    # Plot spike times
    logs = sim.spike_log
    print(logs)
    for label, neuron_id in {
        'x+': net.output_x_plus.id,
        'x-': net.output_x_minus.id,
        'y+': net.output_y_plus.id,
        'y-': net.output_y_minus.id,
        'z+': net.output_z_plus.id,
        'z-': net.output_z_minus.id
    }.items():
        times = logs.get(neuron_id, [])
        plt.plot(times, [label] * len(times), 'o', label=label)

    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron Output")
    plt.title("Lorenz Attractor Output Spikes")
    plt.legend()
    plt.show()
