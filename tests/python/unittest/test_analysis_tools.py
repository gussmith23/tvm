import tvm
from tvm import relay
from tvm.relay import analysis_tools
from tvm.relay.testing import (
    mlp, resnet, dqn, dcgan, mobilenet, lstm, inception_v3, squeezenet, vgg, densenet)


class GetReadableName(analysis_tools.AnalysisPass):
    def visit_call(self, call):
        super().visit_call(call)
        self._add_detail(call, readable_name=call.op.name)


class GetIndex(analysis_tools.AnalysisPass):
    def __init__(self):
        super().__init__()
        self.__id = 0

    def visit_call(self, call):
        super().visit_call(call)
        self._add_detail(call, id=self.__id)
        self.__id += 1


class SummarizeOpTypes(relay.analysis_tools.AnalysisPass):
    def _summarize(self):
        histogram = {}
        for _, data in self._existing_data.items():
            if data['readable_name'] not in histogram:
                histogram[data['readable_name']] = 1
            else:
                histogram[data['readable_name']] += 1
        self._add_summary(histogram)


def test_analysis_tools():
    summaries = {}
    summary_columns = set()
    for (module, _), name in [
        (resnet.get_workload(num_layers=18), 'resnet18'),
        (resnet.get_workload(num_layers=50), 'resnet50'),
        (mobilenet.get_workload(), 'mobilenet'),
        (mlp.get_workload(batch_size=1), 'mlp'),
        (dqn.get_workload(batch_size=1), 'dqn'),
        (dcgan.get_workload(batch_size=1), 'dcgan'),
            # LSTM throws an error w/ analysis framework
            #    (lstm.get_workload(iterations=32, num_hidden=32), 'lstm'),
        (inception_v3.get_workload(), 'inception_v3'),
        (squeezenet.get_workload(), 'squeezenet'),
        (vgg.get_workload(batch_size=1), 'vgg'),
        (densenet.get_workload(), 'densenet'),
    ]:
        program = module['main']
        analyses = [GetReadableName(), GetIndex(), SummarizeOpTypes()]
        _, summary_results = relay.analysis_tools.run_analyses(
            program, analyses)
        summary_columns.update(
            relay.analysis_tools.get_summary_columns(summary_results))
        summaries[name] = summary_results

    summary_columns_ordered = (sorted(list(summary_columns)))
    summary_column_names = list(map(lambda t: t[0], summary_columns_ordered))
    summary_records = list(
        map(
            lambda t: (t[0], ) + analysis_tools.summary_to_record(
                summary_columns_ordered, t[1]), summaries.items()))

    assert summary_column_names ==\
        ['add',
         'concatenate',
         'nn.avg_pool2d',
         'nn.batch_flatten',
         'nn.batch_norm',
         'nn.bias_add',
         'nn.conv2d',
         'nn.conv2d_transpose',
         'nn.dense',
         'nn.dropout',
         'nn.global_avg_pool2d',
         'nn.max_pool2d',
         'nn.relu',
         'nn.softmax',
         'reshape',
         'tanh']

    assert summary_records == \
        [('resnet18', 8, None, None, 1, 19, 1, 21, None, 1, None, 1, 1, 18, 1, None, None),
         ('resnet50', 16, None, None, 1, 51, 1, 53,
          None, 1, None, 1, 1, 50, 1, None, None),
         ('mobilenet', None, None, None, 1, 27, 1, 27,
          None, 1, None, 1, None, 27, 1, None, None),
         ('mlp', None, None, None, 1, None, 3, None,
          None, 3, None, None, None, 2, 1, None, None),
         ('dqn', None, None, None, 1, None, 5, 3, None,
          2, None, None, None, 4, None, None, None),
         ('dcgan', None, None, None, None, 3, None,
          None, 4, 1, None, None, None, 4, None, 1, 1),
         ('inception_v3', None, 11, 9, 1, 94, 1, 94,
          None, 1, None, None, 5, 94, 1, None, None),
         ('squeezenet', None, 8, None, 1, None, 26,
          26, None, None, 1, 1, 3, 26, 1, None, None),
         ('vgg', None, None, None, 1, None, 11, 8,
          None, 3, 2, None, 5, 10, 1, None, None),
         ('densenet', None, None, 4, 1, 121, 1, 120, None, 1, None, None, 1, 121, None, None, None)]


if __name__ == '__main__':
    test_analysis_tools()
