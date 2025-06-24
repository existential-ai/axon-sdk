# Synapse Types in Axon

This document outlines the synapse model implemented in Axon, inspired by the STICK (Spike Time Interval Computational Kernel) framework. Synapses in Axon are the primary mechanism for transmitting and transforming spike-encoded values in time.

---

##  1. Overview

Each synapse in Axon connects a **presynaptic neuron** to a **postsynaptic neuron**, applying a time-delayed effect based on its **type**, **weight**, and **delay**.

All synaptic interactions are event-driven and time-resolved, respecting the temporal precision of interval coding.

---

##  2. Supported Synapse Types

Axon supports four distinct types of synaptic interactions:

| Type    | Symbol | Action on Postsynaptic Neuron |
|---------|--------|-------------------------------|
| `V`     | –      | Adds directly to membrane potential: `V ← V + w` |
| `ge`    | –      | Adds constant input current: `ge ← ge + w` |
| `gf`    | –      | Adds to exponentially decaying input current: `gf ← gf + w` |
| `gate`  | ±1     | Activates or deactivates `gf` dynamics: `gate ← gate + w` |

Each synapse includes a **delay** value `d` (in ms), which specifies when the effect reaches the target neuron after a spike.

---

##  3. Synapse Type Descriptions

###  V-Synapse

- **Function**: Direct voltage jump
- **Use Cases**:
  - Trigger immediate spiking
  - Apply inhibition (if `w < 0`)
  - Detect coincidences
- **Equation**:
  ```python
  V ← V + w
```
### ge-Synapse
- **Function**: Adds persistent excitatory current
- **Use Cases**:
  - Provide sustained excitatory drive
  - Enable long-term potentiation (LTP) 
- **Equation**:
  ```python
    ge ← ge + w
    ```

### gf-Synapse
- **Function**: Adds fast decaying excitatory current
- **Use Cases**:
  - Provide rapid excitatory input
  - Enable short-term plasticity (STP)
- **Equation**:
  ```python
    gf ← gf + w
    ```
### gate-Synapse
- **Function**: Toggles gating for `gf` dynamics
- **Use Cases**:
  - Activate or deactivate `gf` input
  - Control temporal dynamics of the neuron
- **Equation**:
  ```python
    gate ← gate + w
    ```

## 4. Synapse Delay
Each synapse has a **delay** parameter `d` (in milliseconds) that specifies how long after the presynaptic spike the effect will be applied to the postsynaptic neuron. This delay allows for precise temporal computation and modeling of biological synaptic transmission.

Every synapse includes a delay d (in milliseconds):

Represents transmission latency

Defines when the effect arrives at the postsynaptic neuron

Supports precise spike scheduling and coordination

## 5. Use In Computation
Synapse types form the basis of STICK-based operations in Axon:
| Operation       | Synapse Type(s) Used |
| --------------- | -------------------- |
| Memory Storage  | `ge`                 |
| Memory Recall   | `V`                  |
| Log/Exp         | `gf` + `gate`        |
| Control Flow    | `gate`, `V`          |
| Spike Synchrony | `V` + delay routing  |

They are composable and programmable, enabling symbolic logic, arithmetic, and learning mechanisms entirely through spike timing.

## 5. Implementation Summary
```python 
Synapse(
    pre_neuron: ExplicitNeuron,
    post_neuron: ExplicitNeuron,
    weight: float,
    delay: float,
    synapse_type: str  # 'V', 'ge', 'gf', or 'gate'
)
```
Events are scheduled based on the synapse type and delay, allowing for precise control over the timing of postsynaptic effects.

Synaptic effects are handled in:
* `elements.py` -> `AbstractNeuron.receive_synaptic_event(...)`
* `events.py` -> `SpikeEventQueue.add_event()`

