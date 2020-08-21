# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Utilities for changing datatypes of models."""
import tvm
import tvm.topi.testing
import numpy as np
import pytest
from numpy.random import MT19937, RandomState, SeedSequence
from tvm import relay
from tvm.relay.testing.inception_v3 import get_workload as get_inception
from tvm.relay.testing.resnet import get_workload as get_resnet
from tvm.relay.testing.layers import batch_norm_infer
from tvm.relay.testing.mobilenet import get_workload as get_mobilenet
from tvm.target.datatype import register, register_min_func, register_op, create_lower_func, lower_ite, lower_call_pure_extern
from tvm.tir.op import call_pure_extern
from nose.tools import nottest

# we use a random seed to generate input_data
# to guarantee stable tests
rs = RandomState(MT19937(SeedSequence(123456789)))

def convert_ndarray(dst_dtype, array):
    """Converts NDArray(s) into the specified datatype"""
    x = relay.var('x', shape=array.shape, dtype=str(array.dtype))
    cast = relay.Function([x], x.astype(dst_dtype))
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        return relay.create_executor('graph').evaluate(cast)(array)

def change_dtype(src, dst, module, params):
    module = relay.frontend.ChangeDatatype(src, dst)(module)
    module = relay.transform.InferType()(module)
    params = {k: convert_ndarray(dst, v) for k, v in params.items()}
    return module, params

def compare(module, input, src_dtype, dst_dtype, rtol, atol, params = {}, target='llvm'):
    module = relay.transform.SimplifyInference()(module)
    ex = relay.create_executor("graph", mod=module)

    correct = ex.evaluate()(*input, **params)

    module, converted_params = change_dtype(src_dtype, dst_dtype, module, params)
    ex = relay.create_executor("graph", mod=module, target=target)
    # converts all inputs to dst_dtype
    x_converted = [convert_ndarray(dst_dtype, arr) for arr in input]

    # Vectorization is not implemented with custom datatypes
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        maybe_correct = ex.evaluate()(*x_converted, **converted_params)
        # currently this only works for comparing single output
        maybe_correct_converted = convert_ndarray(src_dtype, maybe_correct)
    np.testing.assert_allclose(maybe_correct_converted.asnumpy(),
                                correct.asnumpy(),
                                rtol=rtol,
                                atol=atol)

@pytest.fixture(scope="session", autouse=True)
def setup():
    """Set up tests

    Currently, this registers some custom datatypes using the Bring Your
    Own Datatypes framework.
    """

    # To use datatype operations in an external library, you should first load
    # the library containing the datatype implementation:
    # CDLL("libposit.so", RTLD_GLOBAL)
    # In this case, the datatype library we are using is built right into TVM,
    # so we do not need to explicitly load any library.

    # You can pick a code for your datatype arbitrarily, as long as it is
    # greater than 128 and has not already been chosen.

    register("posites2", 131)

    register_op(create_lower_func(
        {
            (32, 32): "FloatToPosit32es2",
            (32, 16): "FloatToPosit16es2",
            (32, 8): 'FloatToPosit8es2',
        }), 
        "Cast", "llvm", "float", "posites2")
    register_op(create_lower_func(
        {
            (32, 32): "Posit32es2ToFloat",
            (16, 32): 'Posit16es2ToFloat',
            (8, 32): 'Posit8es2ToFloat',
        }), 
        "Cast", "llvm", "posites2", "float")
    register_op(create_lower_func(
        {
            (4, 32): 'IntToPosit32es2',
            (4, 16): 'IntToPosit16es2',
            (4, 8): 'IntToPosit8es2'
        }), 
        "Cast", "llvm", "int", "posites2")
    register_op(create_lower_func({
        32: 'Posit32es2Add',
        16: 'Posit16es2Add',
        8: 'Posit8es2Add'
    }), "Add", "llvm", "posites2")
    register_op(create_lower_func({
        32: 'Posit32es2Sub',
        16: 'Posit16es2Sub',
        8: 'Posit8es2Sub'
    }), "Sub", "llvm", "posites2")
    register_op(create_lower_func({
        32: 'FloatToPosit32es2',
        16: 'FloatToPosit16es2',
        8: 'FloatToPosit8es2'
    }), "FloatImm", "llvm", "posites2")
    register_op(create_lower_func({
        32: 'Posit32es2Mul',
        16: 'Posit16es2Mul',
        8: 'Posit8es2Mul'
    }), "Mul", "llvm", "posites2")
    register_op(create_lower_func({
        32: 'Posit32es2Div',
        16: 'Posit16es2Div',
        8: 'Posit8es2Div'
    }), "Div", "llvm", "posites2")
    register_op(create_lower_func({
        32: 'Posit32es2Max',
        16: 'Posit16es2Max',
        8: 'Posit8es2Max'
    }), "Max", "llvm", "posites2")
    register_op(create_lower_func({
        32: 'Posit32es2Sqrt',
        16: 'Posit16es2Sqrt',
        8: 'Posit8es2Sqrt'
    }), "Call", "llvm", "posites2", intrinsic_name="tir.sqrt")
    register_op(lower_ite,
                "Call",
                "llvm",
                "posites2",
                intrinsic_name="tir.if_then_else")
    register_op(lower_call_pure_extern,
                "Call",
                "llvm",
                "posites2",
                intrinsic_name="tir.call_pure_extern")
    register_op(create_lower_func({
        32: 'Posit32es2Exp',
        16: 'Posit16es2Exp',
        8: 'Posit8es2Exp'
    }), "Call", "llvm", "posites2", intrinsic_name="tir.exp")
    register_op(create_lower_func({
        32: 'Posit32es2Log',
        16: 'Posit16es2Log',
        8: 'Posit8es2Log'
    }), "Call", "llvm", "posites2", intrinsic_name="tir.log")
    register_op(create_lower_func({
        32: 'Posit32es2Sigmoid',
        16: 'Posit16es2Sigmoid',
        8: 'Posit8es2Sigmoid'
    }), "Call", "llvm", "posites2", intrinsic_name="tir.sigmoid")
    register_op(create_lower_func({
        32: 'Posit32es2Tanh',
        16: 'Posit16es2Tanh',
        8: 'Posit8es2Tanh'
    }), "Call", "llvm", "posites2", intrinsic_name="tir.tanh")

    def posit_min_func(num_bits):
        # encode raw bit representation
        # min posit is all 1's in binary
        value = np.dtype('int' + str(num_bits)).type(-1)
        dtype = 'custom[posites2]' + str(num_bits)
        func_map = {
            32: 'RawPosit32es2',
            16: 'RawPosit16es2',
            8: 'RawPosit8es2'
        }
        return call_pure_extern(dtype, func_map[num_bits], value)
    register_min_func(posit_min_func, "posites2")


def run_ops(src_dtype, dst_dtype, rtol=1e-7, atol=1e-7):
    """Run the same op, but with two different datatypes"""
    # used for unary ops, first shape in binary ops
    shape1 = (5, 10, 5)
    # second shape for binary ops
    shape2 = (5, )

    def check_unary_op(op, src_dtype, dst_dtype):
        t1 = relay.TensorType(shape1, src_dtype)
        x = relay.var("x", t1)
        z = op(x)
        x_data = rs.rand(*shape1).astype(t1.dtype)

        module = tvm.IRModule.from_expr(relay.Function([x], z))

        compare(module, (x_data, ), src_dtype, dst_dtype, rtol, atol)

    for op in [
            relay.nn.softmax,
            tvm.relay.log,
            tvm.relay.exp,
            tvm.relay.sqrt,
            tvm.relay.rsqrt,
            tvm.relay.sigmoid,
            tvm.relay.tanh,
            relay.nn.relu,
    ]:
        check_unary_op(op, src_dtype, dst_dtype)

    def check_binary_op(opfunc, src_dtype, dst_dtype):
        t1 = relay.TensorType(shape1, src_dtype)
        t2 = relay.TensorType(shape2, src_dtype)
        x = relay.var("x", t1)
        y = relay.var("y", t2)
        z = opfunc(x, y)
        x_data = rs.rand(*shape1).astype(t1.dtype)
        y_data = rs.rand(*shape2).astype(t2.dtype)
        module = tvm.IRModule.from_expr(relay.Function([x, y], z))

        compare(module, (x_data, y_data), src_dtype, dst_dtype, rtol, atol)

    for op in [
            relay.add,
            relay.subtract,
            relay.divide,
            relay.multiply,
    ]:
        check_binary_op(op, src_dtype, dst_dtype)

    # we would like to test tvm_if_then_else
    # but Relay.IfNode is not lowered to this intrinsic,
    # so to keep our tests consistent with relay, we decide to not unit test
    # Note: tvm_if_then_else is tested as part of the mobile_net model

def run_model(get_workload,
              input_shape,
              src_dtype,
              dst_dtype,
              num_classes,
              rtol=1e-4,
              atol=1e-4):
    module, params = get_workload(image_shape=input_shape,
                                  num_classes=num_classes)

    # generate random input with appropriate shape/type
    input = tvm.nd.array(rs.rand(*input_shape).astype(src_dtype))

    compare(module, (input, ), src_dtype, dst_dtype, rtol, atol, params)

def run_conv2d(src_dtype, dst_dtype, rtol=1e-7, atol=1e-4):
    def run_test_conv2d(src_dtype,
                        dst_dtype,
                        scale,
                        dshape,
                        kshape,
                        padding=(1, 1),
                        groups=1,
                        dilation=(1, 1),
                        **attrs):
        x = relay.var("x", shape=dshape, dtype=src_dtype)
        w = relay.var("w", shape=kshape, dtype=src_dtype)
        y = relay.nn.conv2d(x,
                            w,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            **attrs)
        module = tvm.IRModule.from_expr(relay.Function([x, w], y))
        data = rs.uniform(-scale, scale, size=dshape).astype(src_dtype)
        kernel = rs.uniform(-scale, scale,
                                   size=kshape).astype(src_dtype)

        compare(module, (data, kernel), src_dtype, dst_dtype, rtol, atol)

    # depthwise conv2d
    dshape = (1, 32, 18, 18)
    kshape = (32, 1, 3, 3)
    run_test_conv2d(src_dtype,
                    dst_dtype,
                    1,
                    dshape,
                    kshape,
                    padding=(1, 1),
                    channels=32,
                    groups=32,
                    kernel_size=(3, 3))

    # CUDA is disabled for 'direct' schedule:
    # https://github.com/dmlc/tvm/pull/3070#issuecomment-486597553
    # group conv2d
    dshape = (1, 32, 18, 18)
    kshape = (32, 4, 3, 3)
    run_test_conv2d(src_dtype,
                    dst_dtype,
                    1,
                    dshape,
                    kshape,
                    padding=(1, 1),
                    channels=32,
                    groups=8,
                    kernel_size=(3, 3))
    # also group conv2d
    dshape = (1, 32, 18, 18)
    kshape = (64, 1, 3, 3)
    run_test_conv2d(src_dtype,
                    dst_dtype,
                    1,
                    dshape,
                    kshape,
                    padding=(1, 1),
                    channels=64,
                    groups=32,
                    kernel_size=(3, 3))

    # normal conv2d
    dshape = (1, 3, 224, 224)
    kshape = (10, 3, 3, 3)
    run_test_conv2d(src_dtype,
                    dst_dtype,
                    1,
                    dshape,
                    kshape,
                    padding=(1, 1),
                    channels=10,
                    kernel_size=(3, 3))

    # dilated conv2d
    dshape = (1, 3, 18, 18)
    kshape = (10, 3, 3, 3)
    run_test_conv2d(src_dtype,
                    dst_dtype,
                    1,
                    dshape,
                    kshape,
                    padding=(1, 1),
                    channels=10,
                    kernel_size=(3, 3),
                    dilation=(3, 3))


def test_ops():
    run_ops('float32', 'custom[posites2]8', rtol=1, atol=1)
    run_ops('float32', 'custom[posites2]16', rtol=0.01, atol=1)
    run_ops('float32', 'custom[posites2]32')

def test_conv2d():
    run_conv2d('float32', 'custom[posites2]8', rtol=1, atol=1)
    run_conv2d('float32', 'custom[posites2]16', rtol=0.01, atol=1)
    run_conv2d('float32', 'custom[posites2]32')

def test_batchnorm():
    def run_batchnorm(src_dtype, dst_dtype, rtol=1e-4, atol=1e-4):
        shape = (3, 32, 32)
        t = relay.TensorType(shape, src_dtype)
        x = relay.var("x", t)
        bn = batch_norm_infer(data=x, epsilon=2e-5, scale=False, name='bn_x')
        f = relay.Function(relay.analysis.free_vars(bn), bn)

        x_data = rs.rand(*shape).astype(t.dtype)
        module = tvm.IRModule.from_expr(f)

        zero_data = np.zeros((32), 'float32')
        compare(module, (x_data, zero_data, zero_data, zero_data, zero_data), src_dtype, dst_dtype, rtol, atol)

    run_batchnorm('float32', 'custom[posites2]8', rtol=1, atol=1)
    run_batchnorm('float32', 'custom[posites2]16', rtol=0.01, atol=1)
    run_batchnorm('float32', 'custom[posites2]32')

def test_models():
    # Expected posit8 might be faster, but it's not.
    # run_model(get_mobilenet, (3, 224, 224), 'float32', 'custom[posit8]8')
    # run_model(get_mobilenet, (3, 224, 224), 'float32', 'custom[posit32]32')
    # run_model(get_inception, (3, 299, 299), 'float32', 'custom[posit32]32')
    # run_model(get_resnet, (3, 224, 224), 'float32', 'custom[posit32]32')

    # Run cifar-10 sizes to be a little faster...
    run_model(get_mobilenet, (3, 32, 32),
              'float32',
              'custom[posites2]32',
              num_classes=10)
    # runs on the order of minutes...
    # run_model(get_inception, (3, 299, 299),
    #           'float32',
    #           'custom[posites2]32',
    #           num_classes=10)
    # run_model(get_resnet, (3, 32, 32),
    #           'float32',
    #           'custom[posites2]32',
    #           num_classes=10)

if __name__ == "__main__":
    pytest.main([__file__])
