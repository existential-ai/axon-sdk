import heapq
from .elements import ExplicitNeuron

import itertools


class SpikeEvent:
    def __init__(
        self,
        time: float,
        affected_neuron: ExplicitNeuron,
        synapse_type: str,
        weight: float,
    ):
        self.time = time
        self.affected_neuron = affected_neuron
        self.synapse_type = synapse_type
        self.weight = weight

    def __lt__(self, other):
        # Needed since spike events will be used in a heap
        return self.time < other.time


class SpikeEventQueue:
    def __init__(self):
        self.events: list[SpikeEvent] = []

    def add_event(
        self,
        time: float,
        neuron: ExplicitNeuron,
        synapse_type: str,
        weight: float,
    ):
        event = SpikeEvent(
            time=time,
            affected_neuron=neuron,
            synapse_type=synapse_type,
            weight=weight,
        )
        heapq.heappush(self.events, event)

    def pop_events(self, current_time) -> list[SpikeEvent]:
        events = []
        while self.events and self.events[0].time <= current_time:
            events.append(heapq.heappop(self.events))
        return events


class UniqueEvent:
    _counter = itertools.count()

    def __init__(
        self,
        time: float,
    ):
        self.time = time
        self.id = next(UniqueEvent._counter)

    def __lt__(self, other):
        # Needed events will be used in a heap
        return self.time < other.time


class SpikeHitEvent(UniqueEvent):
    def __init__(
        self,
        t: float,
        hitNeuron: ExplicitNeuron,
        synapse_type: str,
        weight: float,
    ):
        super().__init__(time=t)
        self.hitNeuron = hitNeuron
        self.synapse_type = synapse_type
        self.weight = weight


class NeuronResetEvent(UniqueEvent):
    def __init__(self, t: float, resetNeuron: ExplicitNeuron):
        super().__init__(time=t)
        self.resetNeuron = resetNeuron


class CancelableEventQueue:
    """
    The heap holds the ordered time of the events.
    The mapping points a time to its corresponding events.
    """

    def __init__(self):
        self._time_heap = []
        self._events_at_time: dict[float, list[UniqueEvent]] = {}

    def remove(self, event: UniqueEvent) -> None:
        if (
            event.time in self._events_at_time.keys()
            and event in self._events_at_time[event.time]
        ):
            # Note that removing an event does NOT remove that time from the heap (to avoid having to re-heapify)
            self._events_at_time[event.time].remove(event)

    def add_event(self, event: UniqueEvent) -> UniqueEvent:
        if event.time not in self._events_at_time.keys():
            self._events_at_time[event.time] = []
            heapq.heappush(self._time_heap, event.time)
        self._events_at_time[event.time].append(event)
        return event

    def pop(self) -> list[UniqueEvent]:
        if not self._time_heap:
            raise IndexError("Pop from an empty priority queue")
        events = []
        while self._time_heap and len(events) == 0:
            smallest_time = heapq.heappop(self._time_heap)
            events = self._events_at_time[smallest_time]
            del self._events_at_time[smallest_time]
        return events

    def __len__(self) -> int:
        non_empty_times = [
            l for l in self._time_heap if len(self._events_at_time[l]) != 0
        ]
        return len(non_empty_times)
