# Network Composition & Orchestration in Axon

This guide describes how to **compose**, **connect**, and **orchestrate** symbolic SNN structures in Axon using the STICK model—enabling modular designs, reusable components, and seamless simulation-to-hardware workflows.

---

##  1. Modular Network Architecture

Axon encourages defining networks as combinations of **modules**:

- **Atomic modules**: basic units (e.g., `Adder`, `Multiplier`, `MemoryUnit`)
- **Composite modules**: built by connecting atomic or other composite units

Each module exposes:
- **Input ports** (spike sources)
- **Output ports** (spike sinks)
- **Internal logic** (neurons, synapses, gating)

### Example
```python
from axon.networks import Adder, Multiplier
from axon.composition import compose

add = Adder(name='add1')
mul = Multiplier(name='mul1')
net = compose([add, mul],
              connections=[('add1.out', 'mul1.in1'),
                           ('external.x', 'add1.in1'),
                           ('external.y', 'add1.in2'),
                           ('external.z', 'mul1.in2')])
```

##  2. Connection Patterns
Axon supports flexible connection patterns between modules:
- **Direct connections**: link output of one module to input of another
- **Broadcasting**: send output to multiple inputs



