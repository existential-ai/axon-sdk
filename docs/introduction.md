# Axon: The STICK Software Development Kit



The brain encodes information using precise spike timing, not just rates or continuous activations. Inspired by this, **Axon** is a software framework for building, simulating, and compiling **symbolic spiking neural networks (SNNs)** using the **STICK (Spike Time Interval Computational Kernel)** model.

![Axon Architecture](figs/Top-Architecture.png)

Axon provides an end-to-end pipeline for deploying interval-coded SNNs to ultra-low-power neuromorphic hardware, such as the **ADA** chip. It is built for embedded deployment, yet flexible enough for rapid prototyping and evaluation on general-purpose CPUs.

Axon includes:

- A **Python-based simulator** for cycle-accurate emulation of interval-coded symbolic computation.
- A **hardware-aware compiler** that translates modular spiking circuits into a compact binary model format.
- A **runtime API (RT API)** to interface with the ADA coprocessor over TL-UL or SPI.
- Tools for **resource reporting**, cycle estimation, and model profiling.

If you're building symbolic SNNs for embedded inference, control, or cryptographic tasks, Axon bridges software models with neuromorphic execution.

---

## Axon Structure

| Component           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `axon.simulator`    | Spiking network simulator with support for STICK primitives and gating logic |
| `axon.compiler`     | Converts network graphs into optimized model binaries for ADA execution     |
| `axon.utils`        | Data loading, waveform generation, testbench scripting                      |

---

## Requirements

Axon is built in Python and depends on:

- Python ≥ 3.8
- NumPy
- NetworkX
- TQDM
- Matplotlib (for visual debugging)
- PyVCD (optional, for waveform output)

To simulate hardware behavior with Verilator (optional for advanced use):

- Verilator ≥ 5.034
- C++17 toolchain

---

## Installation

Install the SDK from source:

```bash
git clone https://github.com/neucom/axon.git
cd axon
pip install -e .
```

## Example: Multiplication Network
```
from axon.simulator import Simulator
from axon.networks import MultiplierNetwork
from axon.utils import encode_interval

# Encode input spikes
x1_spikes = encode_interval(0.4)
x2_spikes = encode_interval(0.25)

# Build network
net = MultiplierNetwork()
sim = Simulator(net)

# Inject spikes and simulate
sim.inject(x1=x1_spikes, x2=x2_spikes)
sim.run()

# View results
sim.plot_chronogram()
```


## Citation
If you use Axon or ADA in your research, please cite:
```
@misc{axon2025,
  title        = {Axon: A Software Development Kit for Symbolic Spiking Networks},
  author       = {Neucom},
  howpublished = {\url{https://github.com/neucom/axon}},
  year         = {2025}
}
```

## Contact
If you’re working with Axon or STICK-based hardware and want to share your application, request features, or report issues, reach out via GitHub Issues or contact the Neucom team at contact@neucom.ai.

![Neucom Logo](figs/neucom_logo.png)