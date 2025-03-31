from abc import ABC, abstractmethod
from collections import defaultdict
import heapq
from helpers import flatten_nested_list

from typing import Optional


class AbstractNeuron:
    def __init__(self, Vt, tm, tf, Vreset=0):
        """
        Initialize the neuron with given parameters.

        Parameters:
        Vt (float): Spike threshold voltage (mV).
        Vreset (float): Reset voltage after spike (mV).
        tm (float): Membrane time constant (ms).
        tf (float): Time constant for exponential synapses (ms).
        """
        self.Vt = Vt
        self.Vreset = Vreset
        self.tm = tm
        self.tf = tf

        # Initialize state variables
        self.V = Vreset
        self.ge = 0.0
        self.gf = 0.0
        self.gate = 0

    def update(self, dt) -> bool:
        """
        Update the state of the neuron by one timestep.

        Parameters:
        dt (float): The simulation timestep (ms).

        Returns:
        bool: True if the neuron spikes, False otherwise.
        """
        # Update membrane potential based on the differential equations provided
        self.V += dt * (self.ge + self.gate * self.gf) / self.tm

        # Update gf dynamics if gated
        if self.gate:
            self.gf -= dt * (self.gf / self.tf)

        # Check for spike condition (reset happens explicitly later)
        if self.V >= self.Vt:
            return True
        return False

    def receive_synaptic_event(self, synapse_type, weight):
        """
        Update neuron state based on incoming synaptic event.

        Parameters:
        synapse_type (str): Type of synapse ('V', 'ge', 'gf', 'gate').
        weight (float): Synaptic weight to modify neuron state.
        """
        if synapse_type == 'V':
            self.V += weight
        elif synapse_type == 'ge':
            self.ge += weight
        elif synapse_type == 'gf':
            self.gf += weight
        elif synapse_type == 'gate':
            self.gate = 1 if weight > 0 else 0
        else:
            raise ValueError("Unknown synapse type.")


class ExplicitNeuron(AbstractNeuron):
    def __init__(self, Vt:float, tm:float, tf:float, Vreset:float=0, neuron_id:Optional[str]=None):
        super().__init__(Vt, tm, tf, Vreset)
        self.id = neuron_id
        self.spike_times:list[float] = []

    def reset(self):
        self.V = self.Vreset
        self.ge = 0
        self.gf = 0
        self.gate = 0
        

class DataEncoder:
    def __init__(self, Tmin=10.0, Tcod=100.0):
        """
        Initialize the encoder with the given minimum interval and coding time.

        Parameters:
        Tmin (float): Minimum spike interval (ms).
        Tcod (float): Interval duration representing the maximum encoded value (ms).
        """
        self.Tmin = Tmin
        self.Tcod = Tcod
        self.Tmax = Tmin + Tcod

    def encode_value(self, value) -> tuple[float, float]:
        """
        Encode a value into spike times.

        Parameters:
        value (float): The value to encode, expected between 0 and 1.

        Returns:
        tuple: Two spike times representing the encoded value.
        """
        assert value >= 0 and value <= 1
        interval = self.Tmin + value * self.Tcod
        return (0, interval)
    
    def decode_interval(self, spiking_interval) -> float:
        """
        Decode a spikes interval into a value

        Parameters:
        spiking_interval (float): The value to encode, expected between 0 and 1.

        Returns:
        float: The decoded value
        """
        value = (spiking_interval - self.Tmin) / self.Tcod
        return value
    

class SpikingNetworkModule():
    def __init__(self) -> None:
        self._neurons:list[ExplicitNeuron] = []
        self._subnetworks:list[SpikingNetworkModule] = []

    @property
    def neurons(self) -> list[ExplicitNeuron]:
        total_neurons = []
        total_neurons.extend([n.id if n.id else n for n in self._neurons])
        sub_neurons = flatten_nested_list([subnet.neurons for subnet in self._subnetworks])
        total_neurons.extend(sub_neurons)
        return total_neurons

    @neurons.setter
    def neurons(self, new_neurons:list[ExplicitNeuron]) -> None:
        self._neurons = new_neurons

    def add_neuron(self, neuron:ExplicitNeuron) -> None:
        self._neurons.append(neuron)

    def add_subnetwork(self, subnet:'SpikingNetworkModule') -> None:
        self._subnetworks.append(subnet)



class AbstractSpikingNetwork(ABC):
    def __init__(self):
        self.neurons = {}
        self.synapses = []

    def add_neuron(self, neuron_id, neuron):
        self.neurons[neuron_id] = neuron

    def connect_neurons(self, pre_neuron_id, post_neuron_id, synapse_type, weight, delay):
        synapse = Synapse(
            pre_neuron=self.neurons[pre_neuron_id],
            post_neuron=self.neurons[post_neuron_id],
            synapse_type=synapse_type,
            weight=weight,
            delay=delay
        )
        self.synapses.append(synapse)

    def synapses_from(self, neuron_id):
        is_neuron_id = lambda synapse: synapse.pre_neuron.id is neuron_id
        return list(filter(is_neuron_id, self.synapses))

    @abstractmethod
    def simulate(self, simulation_time, dt):
        pass




class Synapse:
    def __init__(self, pre_neuron:ExplicitNeuron, post_neuron:ExplicitNeuron, weight:float, delay:float, synapse_type:str):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.type = synapse_type
        self.weight = weight
        self.delay = delay


class EventQueue:
    def __init__(self):
        self.events = []

    def add_event(self, event_time, neuron_id, synapse_type, weight):
        heapq.heappush(self.events, (event_time, neuron_id, synapse_type, weight))

    def pop_events(self, current_time):
        events = []
        while self.events and self.events[0][0] <= current_time:
            events.append(heapq.heappop(self.events))
        return events