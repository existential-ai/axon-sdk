from axon_sdk.primitives import SpikingNetworkModule, ExplicitNeuron, DataEncoder
from axon_sdk.networks import (
    SignedMultiplierNormNetwork,
    AdderNetwork,
    SignFlipperNetwork,
)
from .scalar import Scalar, OpType, trace

from typing import Optional


class NeuronHeader:
    def __init__(self, plus: ExplicitNeuron, minus: ExplicitNeuron):
        self.plus = plus
        self.minus = minus


class InputTrigger:
    def __init__(self, value: float, norm: float, neuron_header: NeuronHeader):
        assert value / norm <= 1.0, f"Input value outside range [0, {norm}]"
        assert (
            norm <= 100
        ), f"Guardrail: normalization only tested up to 100; {norm} given"

        self.normalized_value = abs(value) / norm

        if value >= 0:
            self.trigger_neuron = neuron_header.plus
        else:
            self.trigger_neuron = neuron_header.minus


class OutputReader:
    def __init__(self, header: NeuronHeader, norm: float):
        self.read_neuron_plus = header.plus
        self.read_neuron_minus = header.minus

        self.normalization = norm


class ExecutionPlan:
    def __init__(
        self,
        net: SpikingNetworkModule,
        triggers: list[InputTrigger],
        reader: OutputReader,
    ):
        self.net = net
        self.input_triggers = triggers
        self.output_reader = reader


class InjectorNetwork(SpikingNetworkModule):
    def __init__(self, encoder: DataEncoder, module_name: Optional[str] = None):
        super().__init__(module_name)

        Vt = 10.0
        tm = 100.0
        tf = 20.0

        self.inject_plus = self.add_neuron(
            Vt=Vt, tf=tf, tm=tm, neuron_name="inject_input_plus"
        )
        self.inject_minus = self.add_neuron(
            Vt=Vt, tf=tf, tm=tm, neuron_name="inject_input_minus"
        )


class Plug:
    def __init__(self, node: Scalar):
        self.label = str(node.data)
        self.neuron_header: Optional[NeuronHeader] = None

    def __repr__(self):
        return f"<Plug: label {self.label}, id{id(self)}>"


class Connection:
    def __init__(self, pre: Plug, post: Plug):
        self.pre = pre
        self.post = post

    def __repr__(self):
        return f"<pre: {self.pre}, post: {self.post}>"


class OpModuleScaffold:
    def __init__(self, optype: OpType, inps: list[Plug], outp: list[Plug]):
        self.optype = optype
        self.inp_plugs = inps
        self.outp_plug = outp
        self.module: Optional[SpikingNetworkModule] = None


class LoadOpModuleScaffold(OpModuleScaffold):
    def __init__(
        self, value: float, optype: OpType, inps: list[Plug], outp: list[Plug]
    ):
        super().__init__(optype, inps, outp)
        self.value = value


def init_plug_dict(nodes: list[Scalar]) -> dict[Scalar, Plug]:
    empty_dict = {}
    for node in nodes:
        empty_dict[node] = Plug(node)

    return empty_dict


def flatten(root: Scalar) -> tuple[list[OpModuleScaffold], list[Connection], Plug]:
    ops: list[OpModuleScaffold] = []
    connections: list[Connection] = []

    nodes, _ = trace(root)
    scalar_to_plug = init_plug_dict(nodes)

    for node in nodes:
        plug_o = [scalar_to_plug[node]]
        plug_i = [Plug(n) for n in node.prev]

        if node.op == OpType.Load:
            new_op = LoadOpModuleScaffold(
                value=node.data, optype=node.op, inps=plug_i, outp=plug_o
            )
        else:
            new_op = OpModuleScaffold(node.op, inps=plug_i, outp=plug_o)

        ops.append(new_op)

        new_connections = [
            Connection(pre=scalar_to_plug[n], post=plug_i[i])
            for i, n in enumerate(node.prev)
        ]

        connections.extend(new_connections)

    return ops, connections, scalar_to_plug[root]


def spawn_stick_module(
    op: OpModuleScaffold, norm: float, encoder=DataEncoder()
) -> tuple[Optional[SpikingNetworkModule], list[NeuronHeader], Optional[NeuronHeader]]:
    in_header = []
    out_header = None
    match op.optype:
        case OpType.Add:
            mod = AdderNetwork(encoder, module_name="adder_mod")
            in_header = []
            in_header.append(NeuronHeader(plus=mod.input1_plus, minus=mod.input1_minus))
            in_header.append(NeuronHeader(plus=mod.input2_plus, minus=mod.input2_minus))
            out_header = NeuronHeader(plus=mod.output_plus, minus=mod.output_minus)

        case OpType.Load:
            mod = InjectorNetwork(encoder, module_name="injector_mod")
            in_header = []
            out_header = NeuronHeader(plus=mod.inject_plus, minus=mod.inject_minus)
            load_op: LoadOpModuleScaffold = op  # type: ignore
            mod.inject_plus.additional_info = f"<LOAD {load_op.value:.2f}>"
            mod.inject_minus.additional_info = f"<LOAD {load_op.value:.2f}>"

        case OpType.Mul:
            mod = SignedMultiplierNormNetwork(
                encoder=encoder, factor=norm, module_name="mul_norm_mod"
            )
            in_header = []
            in_header.append(NeuronHeader(plus=mod.input1_plus, minus=mod.input1_minus))
            in_header.append(NeuronHeader(plus=mod.input2_plus, minus=mod.input2_minus))
            out_header = NeuronHeader(plus=mod.output_plus, minus=mod.output_minus)

        case OpType.Neg:
            mod = SignFlipperNetwork(encoder=encoder, module_name="inv_mod")
            in_header = []
            in_header.append(NeuronHeader(plus=mod.inp_plus, minus=mod.inp_minus))
            out_header = NeuronHeader(plus=mod.outp_plus, minus=mod.outp_minus)

        case _:
            raise Exception(
                f"op was not initialized correctly due to missing or non-supported optype. Op {op}"
            )

    return mod, in_header, out_header


def fill_op_scafold(op: OpModuleScaffold, norm: float) -> None:
    """
    Input 'op' is modified in place.

    The argument 'op' is filled with an instance of the STICK module
    that implements the computation, and its plugs are wired to neurons of that module.
    """
    mod, in_header, out_header = spawn_stick_module(op, norm)

    assert len(op.inp_plugs) == len(
        in_header
    ), "Mismatch in input plugs of op and input channels of STICK module "

    op.module = mod
    for in_plug, header in zip(op.inp_plugs, in_header):
        in_plug.neuron_header = header

    op.outp_plug[0].neuron_header = out_header


def instantiate_stick_modules(
    ops: list[OpModuleScaffold], net: SpikingNetworkModule, norm: float
) -> SpikingNetworkModule:
    for op in ops:
        fill_op_scafold(op, norm)
        if mod := op.module:
            net.add_subnetwork(mod)

    return net


def wire_modules(
    conns: list[Connection], net: SpikingNetworkModule
) -> SpikingNetworkModule:
    Vt = 10.0
    we = Vt
    Tsyn = 1.0

    for conn in conns:
        if (pre_header := conn.pre.neuron_header) and (
            post_header := conn.post.neuron_header
        ):
            net.connect_neurons(pre_header.plus, post_header.plus, "V", we, Tsyn)
            net.connect_neurons(pre_header.minus, post_header.minus, "V", we, Tsyn)

    return net


def build_stick_net(
    flat_ops: list[OpModuleScaffold], flat_connections: list[Connection], norm: float
) -> SpikingNetworkModule:
    net = SpikingNetworkModule(module_name="enclosing_module")
    net = instantiate_stick_modules(flat_ops, net, norm)
    net = wire_modules(flat_connections, net)
    return net


def get_input_triggers(ops: list[OpModuleScaffold], norm: float) -> list[InputTrigger]:
    triggers: list[InputTrigger] = []
    for op in ops:
        if op.optype == OpType.Load and (header := op.outp_plug[0].neuron_header):
            load_op: LoadOpModuleScaffold = op  # type: ignore
            trigger = InputTrigger(value=load_op.value, norm=norm, neuron_header=header)
            triggers.append(trigger)

    return triggers


def get_output_reader(plug: Plug, norm: float) -> Optional[OutputReader]:
    output_reader = None
    if header := plug.neuron_header:
        output_reader = OutputReader(header, norm)
    return output_reader


def compile_computation(root: Scalar, max_range: float) -> ExecutionPlan:
    assert (
        max_range <= 100
    ), "Max. range  > 100 but only tested to work well until 100; Be at your own risk"

    ops, conn, output_plug = flatten(root)

    net = build_stick_net(ops, conn, max_range)
    input_triggers = get_input_triggers(ops, max_range)
    output_reader = get_output_reader(output_plug, max_range)

    if (not output_reader) or len(input_triggers) == 0:
        raise Exception("Compilatior error: couldn't assign input triggers or readers")

    execPlan = ExecutionPlan(net, input_triggers, output_reader)

    return execPlan
