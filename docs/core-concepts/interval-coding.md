# Interval Coding in Axon

This document explains how Axon implements the **interval-based encoding and computation** as defined by the STICK (Spike Time Interval Computational Kernel) model.

---

##  1. Neuron & Synapse Model

Axon uses a simplified **Integrate-and-Fire** neuron model, supporting three synapse types:

- **V-synapses**: instantaneously modify membrane potential (excitatory `w_e = V_t` or inhibitory `w_i = -V_t`)
- **gₑ-synapses**: conductance-based, model temporal integration
- **g_f-synapses**: fast-gated conductance-based

Each synapse includes a configurable delay (≥ `T_syn`, the minimal delay) :contentReference[oaicite:1]{index=1}.

---

## 2. Interval-Based Value Encoding

Values **x ∈ [0,1]** are encoded in the time difference Δt between two spikes:

```text
Δt = T_min + x · T_cod
x   = (Δt − T_min) / T_cod
```
where:
- **T_min**: minimum time difference (e.g., 1 ms)
- **T_cod**: coding interval (e.g., 10 ms)
- **Δt**: time difference between two spikes

## 3. Interval-Based Computation
Spiking networks can be build to process these interval-encoded values. The network dynamics are governed by the synaptic weights and delays, allowing for complex computations based on the timing of spikes.
* Value `x` is represented by timing between spikes Δt.
* Spiking networks manipulate these intervals via synaptic delays, integration, and gating, executing operations like addition, multiplication, and memory.

## 4. Memory & Control Flow Patterns
Axon includes reusable network patterns for symbolic SNN algorithms, such as:

### 4.1 Volatile Memory
* Uses an accumulator neuron (acc) to store value in membrane potential.

* Spike-to-store encodes interval into potential; recall emits output with the same interval once.

