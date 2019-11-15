
import tvm
from tvm import relay
import tvm.relay.analysis_tools


class MyPass(relay.ExprVisitor):
    def visit_call(self, call):
        super().visit_call(call)
        print(call.op)

class GetReadableName(relay.analysis_tools.AnalysisPass):
    def visit_call(self, call):
        super().visit_call(call)
        self._add_detail(call, readable_name=call.op.name)


class GetIndex(relay.analysis_tools.AnalysisPass):
    def __init__(self):
        super().__init__()
        self.__id = 0

    def visit_call(self, call):
        super().visit_call(call)
        self._add_detail(call, id=self.__id)
        self.__id += 1

program = relay.const(1) - (relay.var('x') * relay.var('y'))
analyses = [GetReadableName(), GetIndex()]

analysis_results = relay.analysis_tools.run_analyses(program, analyses)

print(analysis_results)
