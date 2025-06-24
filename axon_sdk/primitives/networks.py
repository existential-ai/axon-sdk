"""
Spiking Network Composition
===========================

This module defines the `SpikingNetworkModule` class, a hierarchical container for building composable STICK-based
spiking networks with neuron and subnetwork modularity.

Key components:
- `SpikingNetworkModule`: Base class for defining networks with neurons and submodules.
- `flatten_nested_list`: Utility to flatten arbitrarily nested lists.
"""

from .elements import Synapse, ExplicitNeuron

from typing import Optional, Self


def flatten_nested_list(nested_list: list) -> list:
    """
    Recursively flattens an arbitrarily nested list into a single list.

    Args:
        nested_list (list): A list which may contain other lists as elements.

    Returns:
        list: A flat list containing all elements in order.
    """
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_nested_list(item))
        else:
            flat_list.append(item)
    return flat_list


class SpikingNetworkModule:
    """
    Base class for constructing hierarchical spiking networks in the STICK model.

    Each module can contain neurons and nested subnetworks, enabling compositional
    construction of larger networks.

    Attributes:
        _neurons (list[ExplicitNeuron]): List of neurons directly in this module.
        _subnetworks (list[SpikingNetworkModule]): Nested submodules.
        _uid (str): Globally unique identifier for this module.
        _instance_count (int): Internal instance index.
    """
    _global_instance_count = 0

    def __init__(self, module_name: Optional[str] = None) -> None:
        """
        Initialize a new spiking network module.

        Args:
            module_name (str, optional): Optional name for this module used in its UID.
        """
        self._neurons: list[ExplicitNeuron] = []
        self._subnetworks: list[Self] = []
        self._instance_count = SpikingNetworkModule._global_instance_count
        if module_name:
            self._uid = f"(m{self.instance_count})_{module_name}"
        else:
            self._uid = f"(m{self.instance_count})"

        SpikingNetworkModule._global_instance_count += 1

    @property
    def uid(self) -> str:
        """
        Returns:
            str: Unique identifier of this module.
        """
        return self._uid

    @property
    def neurons(self) -> list[ExplicitNeuron]:
        """
        Recursively collect all neurons from this module and its submodules.

        Returns:
            list[ExplicitNeuron]: List of all neurons in the hierarchy.
        """
        total_neurons = []
        total_neurons.extend(self._neurons)
        sub_neurons = flatten_nested_list(
            [subnet.neurons for subnet in self._subnetworks]
        )
        total_neurons.extend(sub_neurons)
        return total_neurons

    @property
    def subnetworks(self) -> list[Self]:
        """
         Returns:
             list[SpikingNetworkModule]: Submodules contained in this module.
         """
        return self._subnetworks

    @property
    def instance_count(self) -> int:
        """
        Returns:
            int: Instance index assigned at construction.
        """
        return self._instance_count

    def recurse_neurons_with_module_uid(self) -> list[dict[ExplicitNeuron, str]]:
        """
        Recursively build a list of dictionaries mapping each neuron to its module UID.

        Returns:
            list[dict[ExplicitNeuron, str]]: One dictionary per neuron/module pair.
        """
        total_neurons_with_module = []
        total_neurons_with_module = [
            {neuron: self.uid} for neuron in self.top_module_neurons
        ]
        sub_neurons = flatten_nested_list(
            [subnet.recurse_neurons_with_module_uid() for subnet in self._subnetworks]
        )
        total_neurons_with_module.extend(sub_neurons)
        return total_neurons_with_module

    @property
    def neurons_with_module_uid(self) -> dict[ExplicitNeuron, str]:
        """
        Get a mapping from all neurons in the hierarchy to their parent module UID.

        Returns:
            dict[ExplicitNeuron, str]: Mapping from neuron to module UID.
        """
        dicts = self.recurse_neurons_with_module_uid()
        combined = {}
        for d in dicts:
            combined.update(d)
        return combined

    @property
    def top_module_neurons(self) -> list[ExplicitNeuron]:
        """
        Returns neurons belonging to current module, without taking submodules into account
        """
        return self._neurons

    def add_neuron(
        self,
        Vt: float,
        tm: float,
        tf: float,
        Vreset: float = 0.0,
        neuron_name: Optional[str] = None,
    ) -> ExplicitNeuron:
        """
        Create and add a neuron to this module.

        Args:
            Vt (float): Threshold voltage.
            tm (float): Membrane time constant.
            tf (float): Synaptic decay time constant.
            Vreset (float, optional): Reset voltage after spike. Defaults to 0.0.
            neuron_name (str, optional): Optional name for this neuron.

        Returns:
            ExplicitNeuron: The newly created neuron.
        """

        new_neuron = ExplicitNeuron(
            Vt=Vt,
            tm=tm,
            tf=tf,
            Vreset=Vreset,
            neuron_name=neuron_name,
            parent_mod_id=self.instance_count,
        )
        self._neurons.append(new_neuron)
        return new_neuron

    def add_subnetwork(self, subnet: "SpikingNetworkModule") -> None:
        """
        Add a nested spiking network module.

        Args:
            subnet (SpikingNetworkModule): The submodule to add.
        """
        self._subnetworks.append(subnet)

    def connect_neurons(
        self,
        pre_neuron: ExplicitNeuron,
        post_neuron: ExplicitNeuron,
        synapse_type: str,
        weight: float,
        delay: float,
    ):
        """
        Connect two neurons via a synapse.

        Args:
            pre_neuron (ExplicitNeuron): Presynaptic neuron.
            post_neuron (ExplicitNeuron): Postsynaptic neuron.
            synapse_type (str): Type of synapse ('V', 'ge', 'gf', 'gate', etc.).
            weight (float): Synaptic weight.
            delay (float): Synaptic delay in seconds.
        """
        synapse = Synapse(
            pre_neuron=pre_neuron,
            post_neuron=post_neuron,
            synapse_type=synapse_type,
            weight=weight,
            delay=delay,
        )
        pre_neuron.out_synapses.append(synapse)
