import pytest

from stick_emulator.primitives import (
    CancelableEventQueue,
    SpikeHitEvent,
    NeuronResetEvent,
    ExplicitNeuron
)

from typing import cast


def test_empty_queue():
    queue = CancelableEventQueue()
    neu1 = ExplicitNeuron(Vt=0.0, tm=0.0, tf=0.0)
    event = SpikeHitEvent(t=0.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    queue.add_event(event)
    queue.remove(event)

    assert len(queue.pop()) == 0

def test_empty_queue_many_steps():
    queue = CancelableEventQueue()
    neu1 = ExplicitNeuron(Vt=1, tm=0.0, tf=0.0)
    ev1 = SpikeHitEvent(t=1.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    ev2 = SpikeHitEvent(t=1.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    queue.add_event(ev1)
    queue.add_event(ev2)
    queue.remove(ev1)
    queue.remove(ev2)

    assert len(queue.pop()) == 0


def test_remove_events_dif_time():
    queue = CancelableEventQueue()
    neu1 = ExplicitNeuron(Vt=0.0, tm=0.0, tf=0.0)
    ev1 = SpikeHitEvent(t=1.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    ev2 = SpikeHitEvent(t=1.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    ev3 = SpikeHitEvent(t=2.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    queue.add_event(ev1)
    queue.add_event(ev2)
    queue.add_event(ev3)

    queue.remove(ev1)
    queue.remove(ev2)

    events = queue.pop()
    assert len(events) == 1
    assert events[0] == ev3
    assert isinstance(events[0], SpikeHitEvent)
    event = cast(SpikeHitEvent, events[0])
    assert event.hitNeuron == neu1


def test_remove_events_single_mult_time():
    queue = CancelableEventQueue()
    neu1 = ExplicitNeuron(Vt=0.0, tm=0.0, tf=0.0)
    ev1 = SpikeHitEvent(t=1.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    ev2 = SpikeHitEvent(t=1.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    ev3 = SpikeHitEvent(t=2.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    ev4 = SpikeHitEvent(t=2.0, hitNeuron=neu1, synapse_type="V", weight=0.0)

    queue.add_event(ev1)
    queue.add_event(ev2)
    queue.add_event(ev3)
    queue.add_event(ev4)

    queue.remove(ev1)

    events1 = queue.pop()
    assert len(events1) == 1
    assert events1[0] == ev2
    assert isinstance(events1[0], SpikeHitEvent)
    event = cast(SpikeHitEvent, events1[0])
    assert event.hitNeuron == neu1

    events2 = queue.pop()
    assert len(events2) == 2


def test_order_mult_pop():
    queue = CancelableEventQueue()
    neu1 = ExplicitNeuron(Vt=0.0, tm=0.0, tf=0.0)
    ev5 = SpikeHitEvent(t=5.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    ev3 = SpikeHitEvent(t=3.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    ev6 = SpikeHitEvent(t=6.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    queue.add_event(ev5)
    queue.add_event(ev3)
    queue.add_event(ev6)

    assert queue.pop()[0] == ev3
    assert queue.pop()[0] == ev5
    assert queue.pop()[0] == ev6


def test_several_events_per_time():
    queue = CancelableEventQueue()
    neu1 = ExplicitNeuron(Vt=0.0, tm=0.0, tf=0.0)

    ev1 = SpikeHitEvent(t=1.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    ev2 = SpikeHitEvent(t=1.0, hitNeuron=neu1, synapse_type="V", weight=0.0)

    ev3 = SpikeHitEvent(t=2.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    ev4 = SpikeHitEvent(t=2.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    ev5 = SpikeHitEvent(t=2.0, hitNeuron=neu1, synapse_type="V", weight=0.0)

    ev6 = SpikeHitEvent(t=3.0, hitNeuron=neu1, synapse_type="V", weight=0.0)

    queue.add_event(ev4)
    queue.add_event(ev3)
    queue.add_event(ev6)
    queue.add_event(ev1)
    queue.add_event(ev2)
    queue.add_event(ev5)

    events = queue.pop()
    assert len(events) == 2
    assert ev1 in events
    assert ev2 in events
    
    events = queue.pop()
    assert len(events) == 3
    assert ev3 in events
    assert ev4 in events
    assert ev5 in events

    events = queue.pop()
    assert len(events) == 1
    assert ev6 in events

def test_len_empty_queue():
    queue = CancelableEventQueue()
    neu1 = ExplicitNeuron(Vt=0.0, tm=0.0, tf=0.0)

    ev1 = SpikeHitEvent(t=1.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    ev2 = SpikeHitEvent(t=2.0, hitNeuron=neu1, synapse_type="V", weight=0.0)
    ev3 = SpikeHitEvent(t=3.0, hitNeuron=neu1, synapse_type="V", weight=0.0)

    queue.add_event(ev2)
    queue.add_event(ev1)
    queue.add_event(ev3)

    queue.remove(ev1)
    queue.remove(ev2)
    queue.remove(ev3)

    assert len(queue) == 0



test_len_empty_queue()