from axon_sdk.primitives import DataEncoder
from axon_sdk.simulator import Simulator

from axon_sdk.compilation import Scalar, compile_computation


if __name__ == "__main__":
    # 1. Computation
    x = Scalar(2.0)
    y = Scalar(3.0)
    z = Scalar(4.0)

    out = (x + y) * z

    out.draw_comp_graph(outfile='basic_computation_graph')

    # 2. Compile
    norm = 100
    execPlan = compile_computation(root=out, max_range=norm)

    # 3. Simulate
    enc = DataEncoder()
    sim = Simulator.init_with_plan(execPlan, enc)
    sim.simulate(simulation_time=600)

    # 4. Readout
    spikes_plus = sim.spike_log.get(execPlan.output_reader.read_neuron_plus.uid, [])
    spikes_minus = sim.spike_log.get(execPlan.output_reader.read_neuron_minus.uid, [])

    if len(spikes_plus) == 2:
        decoded_val = enc.decode_interval(spikes_plus[1] - spikes_plus[0])
        re_norm_value = decoded_val * 100
        print("Received plus output")
        print(f"{re_norm_value}")

    if len(spikes_minus) == 2:
        decoded_val = enc.decode_interval(spikes_minus[1] - spikes_minus[0])
        re_norm_value = -1 * decoded_val * 100
        print("Received minus output")
        print(f"{re_norm_value}")
