"""
Spike Event Scheduling
======================

Defines event queue infrastructure for managing timed spike events in STICK-based spiking neural networks.

Classes:
    - SpikeEvent: Represents a scheduled synaptic event.
    - SpikeEventQueue: Priority queue for time-ordered spike events.
"""

import heapq
from .elements import ExplicitNeuron


class SpikeEvent:
    """
        Represents a scheduled synaptic event in the network.

        Attributes:
            time (float): Simulation time at which the event should occur.
            affected_neuron (ExplicitNeuron): Neuron that receives the event.
            synapse_type (str): Type of synaptic interaction (e.g., 'ge', 'gf', 'gate', 'V').
            weight (float): Synaptic weight to apply during the event.
    """
    def __init__(
        self,
        time: float,
        affected_neuron: ExplicitNeuron,
        synapse_type: str,
        weight: float,
    ):
        """
        Initialize a spike event.

        Args:
            time (float): Time at which the event should trigger.
            affected_neuron (ExplicitNeuron): Target neuron to receive the event.
            synapse_type (str): Type of synapse activated.
            weight (float): Synaptic weight applied.
        """
        self.time = time
        self.affected_neuron = affected_neuron
        self.synapse_type = synapse_type
        self.weight = weight

    def __lt__(self, other):
        """
        Comparison operator to allow sorting events in a priority queue.

        Args:
            other (SpikeEvent): Another event to compare.

        Returns:
            bool: True if this event occurs earlier than `other`.
        """
        # Needed since spike events will be used in a heap
        return self.time < other.time


class SpikeEventQueue:
    """
    Priority queue for managing time-sorted spike events.

    Implements event insertion and retrieval of events scheduled up to the current simulation time.
    """
    def __init__(self):
        """
        Initialize an empty spike event queue.
        """
        self.events: list[SpikeEvent] = []

    def add_event(
        self,
        time: float,
        neuron: ExplicitNeuron,
        synapse_type: str,
        weight: float,
    ):
        """
        Add a new spike event to the queue.

        Args:
            time (float): Scheduled time of the event.
            neuron (ExplicitNeuron): Target neuron.
            synapse_type (str): Synapse type.
            weight (float): Weight of the synaptic input.
        """
        event = SpikeEvent(
            time=time,
            affected_neuron=neuron,
            synapse_type=synapse_type,
            weight=weight,
        )
        heapq.heappush(self.events, event)

    def pop_events(self, current_time) -> list[SpikeEvent]:
        """
        Pop all events scheduled to occur up to the current simulation time.

        Args:
            current_time (float): The current time in simulation.

        Returns:
            List[SpikeEvent]: List of events to apply at this timestep.
        """
        events = []
        while self.events and self.events[0].time <= current_time:
            events.append(heapq.heappop(self.events))
        return events
