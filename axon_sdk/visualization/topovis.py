from axon_sdk.primitives import (
    SpikingNetworkModule,
    ExplicitNeuron,
    Synapse,
)

from axon_sdk.visualization.server import start_server


def generate_mapping_neuron_to_net(
    net: SpikingNetworkModule,
) -> dict[ExplicitNeuron, str]:
    """
    Generates a dictionary that maps each neuron to a net uid.

    Neurons in the top module will be assinged the uid of the top module;
    Neurons in submodules of the top module will be assigned the uid of the first submodule.

    IMP: neurons in submodules within submodules will NOT be given the uid of the
    direct submodule that contains them BUT rather of the submodule of net that contains them.
    """
    mapping: dict[ExplicitNeuron, str] = {}
    for n in net.top_module_neurons:
        mapping[n] = net.uid

    for subnet in net.subnetworks:
        for n in subnet.neurons:
            mapping[n] = subnet.uid

    return mapping


def get_neurons_and_synapses_to_display(
    net: SpikingNetworkModule,
) -> tuple[list[ExplicitNeuron], list[Synapse]]:
    neurons_to_display: set[ExplicitNeuron] = set()
    synapses_to_display: set[Synapse] = set()
    for neu in net.top_module_neurons:
        for syn in neu.out_synapses:
            neurons_to_display.add(syn.pre_neuron)
            neurons_to_display.add(syn.post_neuron)
            synapses_to_display.add(syn)

    mapping_neuron_to_net = generate_mapping_neuron_to_net(net)
    syn_changes_module = (
        lambda n1, n2: mapping_neuron_to_net[n1] != mapping_neuron_to_net[n2]
    )
    # could iter instead on net.neurons - net.top_module_neurons, but not worth the changes
    for neu in net.neurons:

        for syn in neu.out_synapses:
            if syn_changes_module(neu, syn.post_neuron):
                neurons_to_display.add(neu)
                neurons_to_display.add(syn.post_neuron)
                synapses_to_display.add(syn)

    return list(neurons_to_display), list(synapses_to_display)


def get_groups_to_display(
    net: SpikingNetworkModule, neurons_to_display: list[ExplicitNeuron]
) -> list[tuple[ExplicitNeuron, str]]:
    """
    Submodules of net will be displayed as boxes in the visualization.
    To do so, the displayed neurons are assigned uid of the module they belong to. Only neurons belonging
    strictly to a submodule of net are assigned to a group (the top module neurons are not given a group)
    """

    mapping_neu_to_module = generate_mapping_neuron_to_net(net)
    selected_groups = [(n, mapping_neu_to_module[n]) for n in neurons_to_display]
    fiter_top_mod_neurons = lambda group: group[1] != net.uid

    return list(filter(fiter_top_mod_neurons, selected_groups))


def format_nodes(neurons: list[ExplicitNeuron]) -> list[dict[str, str]]:
    nodes = []
    for neuron in neurons:
        uid = neuron.uid
        add_info = neuron.additional_info
        item = {}
        item["id"] = uid
        item["label"] = uid + (f"\n {add_info}" if add_info else "")
        nodes.append(item)

    return nodes


def color_for_synapse(synapse_type: str) -> str:
    if synapse_type == "V":
        return "#000000"
    if synapse_type == "ge":
        return "#FF0830"
    if synapse_type == "gf":
        return "#006400"
    if synapse_type == "gate":
        return "#0E1AFE"
    else:
        return "#000000"


def format_edges(synapses: list[Synapse]) -> list[dict[str, str]]:
    edges = []
    for syn in synapses:
        item = {}
        item["source"] = syn.pre_neuron.uid
        item["target"] = syn.post_neuron.uid
        item["label"] = f"({syn.weight:.3f}; {syn.delay:.3f})"
        item["color"] = color_for_synapse(syn.type)
        item["uid"] = syn.uid
        edges.append(item)
    return edges


def format_groups(groups: list[tuple[ExplicitNeuron, str]]) -> list[dict[str, str]]:
    """
    The input groups contain a list of tuples, where each component contains a neuron
    and the uid of their immediately superior module
    """

    def add_element(d, key, value):
        if key not in d:
            d[key] = []
        d[key].append(value)

    groupped_nodes: dict[str, list[ExplicitNeuron]] = {}
    for group in groups:
        add_element(groupped_nodes, group[1], group[0])

    formatted_groups = []
    for key in groupped_nodes:
        formatted_group = {}
        formatted_group["id"] = key
        formatted_group["label"] = key
        formatted_group["nodes"] = [neuron.uid for neuron in groupped_nodes[key]]
        formatted_groups.append(formatted_group)

    return formatted_groups


def vis_topology(net: SpikingNetworkModule) -> None:
    neurons_to_display, synapses_to_display = get_neurons_and_synapses_to_display(net)
    groups_to_display = get_groups_to_display(net, neurons_to_display)

    nodes = format_nodes(neurons_to_display)
    edges = format_edges(synapses_to_display)
    groups = format_groups(groups_to_display)

    graph_data = {}
    graph_data["nodes"] = nodes
    graph_data["edges"] = edges
    graph_data["groups"] = groups

    start_server(graph_data)
