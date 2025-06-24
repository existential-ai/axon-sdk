# Data Encoding in Axon

The `DataEncoder` class is responsible for converting scalar values into **inter-spike intervals (ISIs)** and decoding them back. This functionality is central to how Axon implements symbolic computation using the **STICK (Spike Time Interval Computational Kernel)** framework.

---

##  1. Concept

In STICK-based networks, numerical values are encoded not in voltage amplitude or spike rate, but in the **time difference between two spikes**. This allows for:

- High temporal resolution
- Symbolic logic without rate coding
- Hardware-friendly encoding (e.g., for ADA)

---

##  2. Encoding Equation

A normalized value \( x \n [0, 1] \) is encoded as a spike interval:

```math
Δt = Tmin + x * Tcod
```
where:
- \( Tmin \) is the minimum interval (e.g., 1 ms
- \( Tcod \) is the coding range (e.g., 100 ms)
- \( dt \) is the resulting inter-spike interval (ISI)

### Example:
```python
Δt = 10 + 0.4 * 100 = 50 ms
```

## Decoding Equation
To decode a spike interval back into a value:
```python
x = (interval - Tmin) / Tcod
```
where:
- `interval` is the time difference between two spikes.
This is the inverse of the encoder.


##  3. DataEncoder Class
The `DataEncoder` class provides methods for encoding and decoding values:

```python
class DataEncoder:
    def __init__(self, Tmin=10.0, Tcod=100.0):
        ...

    def encode_value(self, value: float) -> tuple[float, float]:
        ...

    def decode_interval(self, spiking_interval: float) -> float:
        ...
```

### Attributes:
| Attribute | Description                                         |
| --------- | --------------------------------------------------- |
| `Tmin`    | Minimum ISI (typically 10 ms)                       |
| `Tcod`    | Duration over which \[0,1] is scaled (e.g., 100 ms) |
| `Tmax`    | `Tmin + Tcod` (maximum possible ISI)                |


## Integration
The DataEncoder is used during simulation and output processing:

```python
from axon_sdk.simulator import Simulator
from axon_sdk.encoders import DataEncoder

encoder = DataEncoder(Tmin=10.0, Tcod=100.0)
spike_pair = encoder.encode_value(0.6)  # returns (0.0, 70.0)
```
This spike pair can then be injected into the network, and the simulator will handle the timing based on the encoded intervals.
```python
sim.apply_input_value(value=0.6, neuron=input_neuron)
```

## Output Decoding
```python
interval = spike2_time - spike1_time
decoded_value = encoder.decode_interval(interval)
```

