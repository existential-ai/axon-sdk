"""
Axon SDK
========

Axon is a Python-based simulation toolkit for spike-timing-based computation using the STICK (Spike Timing Interval Computational Kernel) model.

It provides:
- Primitives for defining STICK neurons and synapses.
- Tools for composing symbolic spiking networks.
- Compilation routines to map symbolic models to executable primitives.
- A simulator for event-driven execution and visualization of spike dynamics.

This SDK is part of the Neucom platform developed by Neucom ApS.
"""


from .simulator import Simulator
