from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import get_by_name, qonnx_make_model

from finn.core.onnx_exec import execute_onnx


class DNNContainer(CustomOp):
    def get_nodeattr_types(self):
        return {
            "body": ("g", True, ""),
            "io_map": ("s", True, "{}"),
        }

    def get_nodeattr(self, name):
        # Copied from FinnLoop OP
        try:
            (dtype, req, def_val, allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if dtype == "g":
                    ret = attr.__getattribute__(dtype)
                    ret = ModelWrapper(qonnx_make_model(ret))
                    return ret
                else:
                    return super().get_nodeattr(name)
            else:
                if req:
                    raise Exception(
                        """Required attribute %s unspecified in
                    a %s node"""
                        % (name, self.onnx_node.op_type)
                    )
                else:
                    # not set, return default value
                    return def_val
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name)

    def set_nodeattr(self, name, value):
        # Copied from FinnLoop OP
        try:
            (dtype, req, def_val, allowed_values) = self.get_nodeattr_def(name)
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # g : graph
                if dtype == "g":
                    attr.g.CopyFrom(value.graph)
                else:
                    super().set_nodeattr(name, value)
            else:
                super().set_nodeattr(name, value)
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name)

    def make_shape_compatible_op(self, model):
        pass

    def infer_node_datatype(self, model):
        pass

    def execute_node(self, context, graph):
        # Copied from GenericPartition
        # Validate this code
        model = self.get_nodeattr("body")
        return_full_exec_context = 1
        node = self.onnx_node
        inp_ctx = dict(filter(lambda x: x[0] in node.input, context.items()))
        # inputs may have been renamed in partition
        for i, old_iname in enumerate(node.input):
            new_iname = model.graph.input[i].name
            if old_iname != new_iname:
                inp_ctx[new_iname] = inp_ctx[old_iname]
                del inp_ctx[old_iname]
        ret = execute_onnx(model, inp_ctx, return_full_exec_context)
        # outputs may have been renamed in partition
        for i, node_oname in enumerate(node.output):
            model_oname = model.graph.output[i].name
            context[node_oname] = ret[model_oname]
        # prefix and insert exec context entries
        if return_full_exec_context:
            for tname in ret.keys():
                if tname not in [x.name for x in model.graph.output]:
                    context[node.name + "_" + tname] = ret[tname]

    def verify_node(self):
        # Copied from GenericPartition
        info_messages = []

        # verify number of attributes
        num_of_attr = 2
        if len(self.onnx_node.attribute) == num_of_attr:
            info_messages.append("The number of attributes is correct")
        else:
            info_messages.append(
                """The number of attributes is incorrect,
            {} should have {} attributes""".format(
                    self.onnx_node.op_type, num_of_attr
                )
            )
        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("body")
            self.get_nodeattr("io_map")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The necessary attributes do not exist.
                DNNContainer needs the following attribute(s):
                body, io_map"""
            )

        return info_messages
