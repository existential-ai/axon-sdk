# Compilation (`compilation/compiler.py`)

The `compiler.py` module is responsible for transforming a **high-level symbolic computation** expressed as a graph of `Scalar` operations into an **executable STICK network** composed of spiking neurons and synaptic connections. The resulting structure can be simulated, visualized, or deployed on neuromorphic hardware such as Ada. 

The compiler performs the following steps:

1. **Flatten** a scalar expression into nodes and connections
2. **Map each operation** (`Add`, `Mul`, `Neg`, `Load`) to a pre-built STICK subnetwork
3. **Wire these modules** together via excitatory synapses
4. **Generate input triggers** and output readers
5. Return an `ExecutionPlan` for simulation or deployment

## Compilation Pipeline
The compilation process consists of several key stages:

Scalar Expression
↓
trace() + flatten()
↓
OpModuleScaffold list + Connection list
↓
spawn_stick_module() for each op
↓
fill_op_scafold() binds plugs to neurons
↓
instantiate_stick_modules() adds subnetworks
↓
wire_modules() connects neuron headers
↓
get_input_triggers() + get_output_reader()
↓
ExecutionPlan


---

##  Core Components

###  `OpModuleScaffold`
Represents one computation node (e.g. add, mul, load). Stores:
- Operation type (`OpType`)
- Input/output plugs
- Pointer to associated STICK module

###  `Plug`
A named handle to a specific scalar value or intermediate result. Later linked to neuron headers.

###  `InputTrigger`
Encodes a scalar input value (normalized to [0, 1]) and identifies the `plus` or `minus` neuron to inject.

###  `OutputReader`
Defines how to decode spike intervals from output neuron headers into scalar values.

###  `ExecutionPlan`
Holds the final network and all I/O mappings:
```python
ExecutionPlan(
    net=SpikingNetworkModule,
    triggers=[InputTrigger, ...],
    reader=OutputReader
)
```

## Major Functions

* `flatten(root: Scalar)`
Flattens the computation graph and wraps nodes as OpModuleScaffold. Tracks dependencies as Connections.

* `spawn_stick_module(op, norm)`
Creates a STICK subnetwork corresponding to the operation:

Add → AdderNetwork

Mul → SignedMultiplierNormNetwork

Load → InjectorNetwork

Neg → SignFlipperNetwork

Returns module + input/output neuron headers.

* `fill_op_scafold(op)`
Populates the scaffold:

Assigns STICK module to op.module

Binds input/output Plugs to NeuronHeaders

* `instantiate_stick_modules(ops, net, norm)`
Instantiates and attaches all submodules to the main SpikingNetworkModule.

* `wire_modules(conns, net)`
Adds V-synapse connections between output and input neurons across modules.

* `get_input_triggers(ops)`
Extracts Load operations and creates corresponding InputTriggers.

* `get_output_reader(plug)`
Identifies the neuron header from the final output and wraps it into an OutputReader

## Final Step: `compile_computation(root: Scalar, max_range: float)`
This function drives the full compilation process:

1. Flattens the symbolic expression

2. Builds and wires STICK modules

3. Extracts inputs and outputs

4. Returns an ExecutionPlan


```python
plan = compile_computation(root=my_expr, max_range=50.0)
```

### Example 
```python
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
```
