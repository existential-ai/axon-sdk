"""
Neuron and Synapse Primitives
=============================

Defines core primitives for STICK-based spiking networks:
    - `AbstractNeuron`: A base class implementing neuron dynamics and synaptic event integration.
    - `ExplicitNeuron`: A concrete subclass with spike tracking and output synapses.
    - `Synapse`: Represents a connection between two neurons with a type, weight, and delay.
"""

from .helpers import flatten_nested_list

from typing import Optional


class AbstractNeuron:
    """
    Base class for neurons in the STICK model.

    Implements gated synaptic integration and threshold-based spike logic.

    Attributes:
        Vt (float): Spike threshold voltage.
        Vreset (float): Voltage after reset.
        tm (float): Membrane time constant.
        tf (float): Time constant for 'gf' decay.
        V (float): Membrane potential.
        ge (float): Excitatory conductance.
        gf (float): Gated conductance.
        gate (float): Gating input level.
        uid (str): Unique identifier for neuron.
    """
    _instance_count = 0

    def __init__(
        self,
        Vt,
        tm,
        tf,
        Vreset=0.0,
        neuron_name: Optional[str] = None,
        parent_mod_id: Optional[str] = None,
        additional_info: Optional[str] = None
    ):
        """
        Initialize a neuron with specified model parameters.

        Args:
            Vt (float): Threshold voltage for spiking.
            tm (float): Membrane time constant.
            tf (float): Synaptic decay time constant.
            Vreset (float, optional): Voltage after spike reset. Defaults to 0.0.
            neuron_name (str, optional): Human-readable name for tracing.
            parent_mod_id (str, optional): Module ID this neuron belongs to.
            additional_info (str, optional): Debugging or display metadata.
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

        if parent_mod_id is not None:
            self._uid = (
                f"(m{parent_mod_id},n{AbstractNeuron._instance_count})_{neuron_name}"
            )
        else:
            self._uid = f"(n{AbstractNeuron._instance_count})_{neuron_name}"

        self.additional_info = additional_info
        AbstractNeuron._instance_count += 1

    @property
    def uid(self) -> str:
        """
        Returns:
            str: Unique identifier of this neuron instance.
        """
        return self._uid

    def update_and_spike(self, dt) -> tuple[float, bool]:
        """
        Update the neuron's membrane potential and internal state.

        Args:
            dt (float): Time increment in milliseconds.

        Returns:
            tuple[float, bool]: (Updated voltage, Spike flag).
        """
        spike = False
        # Update membrane potential based on the differential equations provided
        self.V += dt * (self.ge + self.gate * self.gf) / self.tm

        # Update gf dynamics if gated
        if self.gate:
            self.gf -= dt * (self.gf / self.tf)

        # Check for spike condition (reset happens explicitly later)
        if self.V >= self.Vt:
            spike = True
        return (self.V, spike)

    def receive_synaptic_event(self, synapse_type, weight):
        """
        Apply a synaptic event to update internal state variables.

        Args:
            synapse_type (str): Type of synapse ('V', 'ge', 'gf', 'gate').
            weight (float): Synaptic weight.

        Raises:
            ValueError: If synapse type is not recognized.
        """
        if synapse_type == "V":
            self.V += weight
        elif synapse_type == "ge":
            self.ge += weight
        elif synapse_type == "gf":
            self.gf += weight
        elif synapse_type == "gate":
            self.gate += weight
        else:
            raise ValueError("Unknown synapse type.")


class ExplicitNeuron(AbstractNeuron):
    """
    A fully defined neuron used in simulations with connection and spike history.

    Attributes:
        spike_times (list[float]): Timestamps of all emitted spikes.
        out_synapses (list[Synapse]): Outgoing synapses from this neuron.
    """
    def __init__(
        self,
        Vt: float,
        tm: float,
        tf: float,
        Vreset: float = 0.0,
        neuron_name: Optional[str] = None,
        parent_mod_id: Optional[str] = None,
        additional_info: Optional[str] = None
    ):
        """
        Initialize an explicit neuron with connectivity and logging.

        Args:
            Vt (float): Spike threshold voltage.
            tm (float): Membrane time constant.
            tf (float): Synaptic decay time constant.
            Vreset (float, optional): Reset voltage after spiking.
            neuron_name (str, optional): Optional neuron label.
            parent_mod_id (str, optional): ID of parent module.
            additional_info (str, optional): Extra display/debugging metadata.
        """
        super().__init__(Vt, tm, tf, Vreset, neuron_name, parent_mod_id, additional_info)
        self.spike_times: list[float] = []
        self.out_synapses: list[Synapse] = []

    def reset(self):
        """
        Reset internal neuron state after a spike.
        """
        self.V = self.Vreset
        self.ge = 0
        self.gf = 0
        self.gate = 0


class Synapse:
    """
    A synaptic connection between two neurons in the network.

    Attributes:
        pre_neuron (ExplicitNeuron): Source neuron.
        post_neuron (ExplicitNeuron): Destination neuron.
        type (str): Synapse type ('V', 'ge', 'gf', or 'gate').
        weight (float): Synaptic weight.
        delay (float): Transmission delay in milliseconds.
        uid (str): Unique synapse ID.
    """
    _instance_count = 0

    def __init__(
        self,
        pre_neuron: ExplicitNeuron,
        post_neuron: ExplicitNeuron,
        weight: float,
        delay: float,
        synapse_type: str,
    ):
        """
        Initialize a synapse between two neurons.

        Args:
            pre_neuron (ExplicitNeuron): Presynaptic neuron.
            post_neuron (ExplicitNeuron): Postsynaptic neuron.
            weight (float): Synaptic weight to apply.
            delay (float): Delay in ms before effect is applied.
            synapse_type (str): Type of synaptic influence.
        """
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.type = synapse_type
        self.weight = weight
        self.delay = delay

        self._uid = f"synapse_{Synapse._instance_count}"
        Synapse._instance_count += 1

    @property
    def uid(self) -> str:
        """
        Returns:
            str: Unique identifier for this synapse.
        """
        return self._uid
