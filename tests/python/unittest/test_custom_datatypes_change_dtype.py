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
import topi.testing
import numpy as np
from tvm import relay
from tvm.relay.testing.inception_v3 import get_workload as get_inception
from tvm.relay.testing.resnet import get_workload as get_resnet
from tvm.relay.testing.mobilenet import get_workload as get_mobilenet
from tvm.relay import transform
from nose.tools import nottest
import ctypes

tgt = "llvm"


def setup():
    """Set up tests

    Currently, this registers some custom datatypes using the Bring Your
    Own Datatypes framework.
    """

    # To use datatype operations in an external library, you should first load
    # the library containing the datatype implementation:
    # CDLL("libmybfloat16.so", RTLD_GLOBAL)
    # In this case, the datatype library we are using is built right into TVM,
    # so we do not need to explicitly load any library.

    # You can pick a code for your datatype arbitrarily, as long as it is
    # greater than 128 and has not already been chosen.

    tvm.datatype.register("posit", 131)

    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("FloatToPosit32es2"), "Cast", "llvm",
        "posit", "float")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("Posit32es2ToFloat"), "Cast", "llvm",
        "float", "posit")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("IntToPosit32es2"),
                             "Cast", "llvm", "posit", "int")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Posit32es2Add"),
                             "Add", "llvm", "posit")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Posit32es2Sub"),
                             "Sub", "llvm", "posit")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_func("FloatToPosit32es2"), "FloatImm",
        "llvm", "posit")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Posit32es2Mul"),
                             "Mul", "llvm", "posit")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Posit32es2Div"),
                             "Div", "llvm", "posit")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Posit32es2Max"),
                             "Max", "llvm", "posit")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Posit32es2Sqrt"),
                             "Call",
                             "llvm",
                             "posit",
                             intrinsic_name="sqrt")
    # TODO(gus) not sure if this will work...
    tvm.datatype.register_op(tvm.datatype.lower_ite,
                             "Call",
                             "llvm",
                             "posit",
                             intrinsic_name="tvm_if_then_else")
    tvm.datatype.register_op(tvm.datatype.create_lower_func("Posit32es2Exp"),
                             "Call",
                             "llvm",
                             "posit",
                             intrinsic_name="exp")
    # TODO(gus) these aren't actually right. these are double min(actually lowest)/max.
    tvm.datatype.register_min_func(lambda num_bits: -1.79769e+308, "posit")

    tvm.datatype.register("datatype_in_python", 132)

    def __datatype_in_python_add(a, b):
        print(a)
        print(b)
        return a + b

    tvm.datatype.register_op(
        tvm.datatype.create_lower_to_python_func(__datatype_in_python_add,
                                                 'llvm', 'Add',
                                                 'datatype_in_python'), "Add",
        "llvm", "datatype_in_python")

    def __datatype_in_python_cast_to_float(value):
        print(value)
        as_c_int32 = ctypes.c_int32(value)
        as_float = ctypes.c_float.from_address(
            ctypes.addressof(as_c_int32)).value
        print("as float: {}".format(as_float))
        return as_float

    def __datatype_in_python_cast_from_float(value):
        print(value)
        as_c_float = ctypes.c_float(value)
        as_int32 = ctypes.c_int32.from_address(
            ctypes.addressof(as_c_float)).value
        print(as_int32)
        return as_int32

    tvm.datatype.register_op(
        tvm.datatype.create_lower_to_python_func(
            __datatype_in_python_cast_to_float, 'llvm', 'Cast', 'float',
            'datatype_in_python'), "Cast", "llvm", 'float',
        "datatype_in_python")
    tvm.datatype.register_op(
        tvm.datatype.create_lower_to_python_func(
            __datatype_in_python_cast_from_float, 'llvm', 'Cast',
            'datatype_in_python', 'float'), "Cast", "llvm",
        "datatype_in_python", 'float')


def convert_ndarray(dst_dtype, array, executor):
    """Converts an NDArray into the specified datatype"""
    x = relay.var('x', shape=array.shape, dtype=str(array.dtype))
    cast = relay.Function([x], x.astype(dst_dtype))
    return executor.evaluate(cast)(array)


def change_dtype(src, dst, expr, params, executor):
    """Change the datatype of a relay expression"""
    # TODO(gus) There's probably a better way to do this---update cdtype to work
    # over modules
    expr = transform.InferType()(relay.Module.from_expr(expr))
    cdtype = relay.frontend.ChangeDatatype(src, dst)
    expr = cdtype.visit(expr['main'])
    expr = transform.InferType()(relay.Module.from_expr(expr))
    expr = expr['main']
    params = dict(
        (p, convert_ndarray(dst, params[p], executor)) for p in params)
    return expr, params


def run_ops(src_dtype, dst_dtype):
    """Run the same op, but with two different datatypes"""
    def check_unary_op(op, src_dtype, dst_dtype):
        t1 = relay.TensorType((5, 10, 5))
        x = relay.var("x", t1)
        z = op(x)
        x_data = np.random.rand(5, 10, 5).astype(t1.dtype)

        func = relay.Function([x], z)

        ex = relay.create_executor("graph")

        correct = ex.evaluate(func)(x_data)

        func, _ = change_dtype(src_dtype, dst_dtype, func, [], ex)

        x_converted = convert_ndarray(dst_dtype, x_data, ex)
        maybe_correct = ex.evaluate(func)(x_converted)
        maybe_correct_converted = convert_ndarray(src_dtype, maybe_correct, ex)
        np.testing.assert_allclose(maybe_correct_converted.asnumpy(),
                                   correct.asnumpy(),
                                   rtol=0.0001,
                                   atol=0.0001)

    for op in [
            # TODO(gus) implement these
            #tvm.relay.log,
            tvm.relay.exp,
            tvm.relay.sqrt,
            tvm.relay.rsqrt,
            # TODO(gus) implement these
            #tvm.relay.sigmoid,
            # TODO(gus) implement these
            #tvm.relay.tanh,
            relay.nn.relu,
            relay.nn.softmax
    ]:
        check_unary_op(op, src_dtype, dst_dtype)

    def check_binary_op(opfunc, src_dtype, dst_dtype):
        t1 = relay.TensorType((5, 10, 5), src_dtype)
        t2 = relay.TensorType((5, ), src_dtype)
        x = relay.var("x", t1)
        y = relay.var("y", t2)
        z = opfunc(x, y)
        x_data = np.random.rand(5, 10, 5).astype(t1.dtype)
        y_data = np.random.rand(5).astype(t2.dtype)
        func = relay.Function([x, y], z)

        ex = relay.create_executor("graph")

        correct = ex.evaluate(func)(x_data, y_data)

        func, _ = change_dtype(src_dtype, dst_dtype, func, [], ex)

        x_converted = convert_ndarray(dst_dtype, x_data, ex)
        y_converted = convert_ndarray(dst_dtype, y_data, ex)

        maybe_correct = ex.evaluate(func)(x_converted, y_converted)
        maybe_correct_converted = convert_ndarray(src_dtype, maybe_correct, ex)
        np.testing.assert_allclose(correct.asnumpy(),
                                   maybe_correct_converted.asnumpy(),
                                   rtol=0.001,
                                   atol=0.001)

    for op in [
            relay.add,
            relay.subtract,
            relay.divide,
            relay.multiply,
    ]:
        check_binary_op(op, src_dtype, dst_dtype)


def run_model(get_workload, input_shape, src_dtype, dst_dtype):
    expr, params = get_workload()

    ex = relay.create_executor("graph")

    # Convert the input into the correct format.
    input = tvm.nd.array(np.random.rand(*input_shape).astype(src_dtype))

    correct = ex.evaluate(expr)(input, **params)

    # Simplifying inference is essential right now, as batch norms (which get
    # removed) are broken with custom datatypes.
    expr = relay.ir_pass.simplify_inference(expr)
    expr, params = change_dtype(src_dtype, dst_dtype, expr, params, ex)

    input = convert_ndarray(dst_dtype, input, ex)

    # Vectorization is not implemented with custom datatypes.
    with tvm.build_config(disable_vectorize=True):
        result = ex.evaluate(expr)(input, **params)
        print(correct)
        print(convert_ndarray(src_dtype, result, ex))

    tvm.testing.assert_allclose(convert_ndarray(src_dtype, result,
                                                ex).asnumpy(),
                                correct.asnumpy(),
                                rtol=0.001,
                                atol=0.001)


def run_conv2d(src_dtype, dst_dtype):
    def run_test_conv2d(src_dtype,
                        dst_dtype,
                        scale,
                        dshape,
                        kshape,
                        padding=(1, 1),
                        fref=None,
                        groups=1,
                        dilation=(1, 1),
                        except_targets=None,
                        **attrs):
        if except_targets is None:
            except_targets = []

        x = relay.var("x", shape=dshape, dtype=src_dtype)
        w = relay.var("w", dtype=src_dtype)
        y = relay.nn.conv2d(x,
                            w,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            **attrs)
        func = relay.Function([x, w], y)
        data = np.random.uniform(-scale, scale, size=dshape).astype(src_dtype)
        kernel = np.random.uniform(-scale, scale,
                                   size=kshape).astype(src_dtype)
        dkernel = topi.testing.dilate_python(kernel, (1, 1) + dilation)
        if fref is None:
            ref_res = topi.testing.conv2d_nchw_python(
                data.astype(src_dtype),
                dkernel.astype(src_dtype),
                1,
                padding,
                groups=groups)
        else:
            ref_res = fref(data.astype(src_dtype), dkernel.astype(src_dtype))

        for target, ctx in [("llvm", tvm.cpu(0))]:
            if target in except_targets:
                continue
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            # convert function
            func, _ = change_dtype(src_dtype, dst_dtype, func, [], intrp1)
            data_converted = convert_ndarray(dst_dtype, data, intrp1)
            kernel_converted = convert_ndarray(dst_dtype, kernel, intrp1)
            with tvm.build_config(disable_vectorize=True):
                op_res1 = intrp1.evaluate(func)(data_converted,
                                                kernel_converted)
            op_res1_converted = convert_ndarray(src_dtype, op_res1, intrp1)
            # TODO(gus) previous rtol, atol 1e-5
            tvm.testing.assert_allclose(op_res1_converted.asnumpy(),
                                        ref_res,
                                        rtol=0.5,
                                        atol=0.5)

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
                    kernel_size=(3, 3),
                    fref=lambda x, w: topi.testing.
                    depthwise_conv2d_python_nchw(x, w, (1, 1), "SAME"))

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
                    kernel_size=(3, 3),
                    except_targets=['cuda'])
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
                    kernel_size=(3, 3),
                    except_targets=['cuda'])

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
    run_ops('float32', 'custom[posit]32')


# disabled for now, because it's slow
@nottest
def test_conv2d():
    run_conv2d('float32', 'custom[posit]32')


# disabled for now, because it's slow
@nottest
def test_models():
    run_model(get_mobilenet, (3, 224, 224), 'float32', 'custom[posit]32')
    run_model(get_inception, (3, 299, 299), 'float32', 'custom[posit]32')
    run_model(get_resnet, (3, 224, 224), 'float32', 'custom[posit]32')


def test_datatype_in_python():
    t1 = relay.TensorType((5, 10, 5))
    x = relay.var("x", t1)
    y = relay.var("y", t1)
    add = x + y
    func = relay.Function([x, y], add)

    x_data = np.random.rand(5, 10, 5).astype(t1.dtype)
    y_data = np.random.rand(5, 10, 5).astype(t1.dtype)

    ex = relay.create_executor("graph")

    correct = ex.evaluate(func)(x_data, y_data)

    func, _ = change_dtype("float32", "custom[datatype_in_python]32", func, [],
                           ex)

    x_converted = convert_ndarray("custom[datatype_in_python]32", x_data, ex)
    import pdb
    pdb.set_trace()
    y_converted = convert_ndarray("custom[datatype_in_python]32", y_data, ex)

    maybe_correct = ex.evaluate(func)(x_converted, y_converted)


def test_datatype_in_python_simple_cast():
    t = relay.TensorType((1, 1), "float32")
    x = relay.var("x", t)
    program = x.astype("custom[datatype_in_python]32")
    program = program.astype("float32")
    program = relay.Function([x], program)
    x_data = np.random.rand(1, 1).astype(t.dtype)
    ex = relay.create_executor("graph")
    np.testing.assert_equal(x_data, ex.evaluate(program)(x_data).asnumpy())


if __name__ == "__main__":
    setup()
    #test_ops()
    # These all run very slowly:
    # test_conv2d()
    # test_models()
    #test_datatype_in_python()
    test_datatype_in_python_simple_cast()
