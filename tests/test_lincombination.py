import pytest

from axon_sdk.networks import LinearCombinatorNetwork
from axon_sdk.primitives import DataEncoder
from axon_sdk.simulator import Simulator


@pytest.mark.parametrize(
    "coeffs, inputs", [
        ([1.0, 1.0, 1.0], [0.1, 0.1, 0.1]),
        ([0.5, 0.5, 0.5], [0.1, 0.1, 0.1]),
        ([0.9, 0.9, 0.9], [0.3, 0.3, 0.3]),
    ]
)
def test_mul(coeffs: list[float], inputs: list[float]):
    assert len(coeffs) == len(inputs), f"Mismatch len between coeffs and inputs"

    encoder = DataEncoder(Tmin=10.0, Tcod=100.0)

    net = LinearCombinatorNetwork(encoder, N=len(coeffs), coeff=coeffs)
    sim = Simulator(net, encoder)
    for idx, inp_val in enumerate(inputs):
        if inp_val >= 0:
            sim.apply_input_value(abs(inp_val), net.input_plus[idx], t0=0)
        else:
            sim.apply_input_value(abs(inp_val), net.input_minus[idx], t0=0)

    sim.simulate(400)

    spikes_plus = sim.spike_log.get(net.output_plus.uid, [])
    spikes_minus = sim.spike_log.get(net.output_minus.uid, [])

    actual_result = sum(c * i for c, i in zip(coeffs, inputs))
    if actual_result > 0:
        assert len(spikes_plus) == 2, f"Expected 2 spikes in + output, got {len(spikes_plus)}"
        assert len(spikes_minus) == 0, f"Expected 0 spikes in - output, got {len(spikes_minus)}"

        interval = spikes_plus[1] - spikes_plus[0]
        decoded_result = encoder.decode_interval(interval)

    elif actual_result < 0:
        assert len(spikes_minus) == 2, f"Expected 2 spikes in - output, got {len(spikes_minus)}"
        assert len(spikes_plus) == 0, f"Expected 0 spikes in + output, got {len(spikes_plus)}"

        interval = spikes_minus[1] - spikes_minus[0]
        decoded_result = -1 * encoder.decode_interval(interval)
                     
                    
    assert actual_result == pytest.approx(decoded_result, abs=1e-2), f"Expected output {actual_result}, got {decoded_result}"