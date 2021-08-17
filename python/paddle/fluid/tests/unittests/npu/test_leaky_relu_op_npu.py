#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import paddle
import paddle.nn.functional as F
import paddle.fluid as fluid
import sys
sys.path.append("..")
from op_test import OpTest

import numpy as np
import unittest

paddle.enable_static()
SEED = 1024


def ref_leaky_relu(x, alpha=0.01):
    out = np.copy(x)
    out[out < 0] *= alpha
    return out


class TestLeakyRelu(OpTest):
    def get_alpha(self):
        return 0.02

    def setUp(self):
        self.set_npu()
        self.op_type = "leaky_relu"
        self.init_dtype()
        alpha = self.get_alpha()

        np.random.seed(SEED)
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.05
        out = ref_leaky_relu(x, alpha)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'alpha': alpha}

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ['X'], 'Out')

    def init_dtype(self):
        self.dtype = np.float32


class TestLeakyReluAlpha1(TestLeakyRelu):
    def get_alpha(self):
        return 2


class TestLeakyReluAlpha2(TestLeakyRelu):
    def get_alpha(self):
        return -0.01


class TestLeakyReluAlpha3(TestLeakyRelu):
    def get_alpha(self):
        return -2.0


class TestLeakyReluFP16(TestLeakyRelu):
    def init_dtype(self):
        self.dtype = np.float16


class TestLeakyReluAPI(unittest.TestCase):
    # test paddle.nn.LeakyReLU, paddle.nn.functional.leaky_relu,
    # fluid.layers.leaky_relu
    def setUp(self):
        np.random.seed(SEED)
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype('float32')
        self.place=paddle.NPUPlace(0) if paddle.is_compiled_with_npu() \
            else paddle.CPUPlace()

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', [10, 12])
            out1 = F.leaky_relu(x)
            m = paddle.nn.LeakyReLU()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_leaky_relu(self.x_np)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.leaky_relu(x)
        m = paddle.nn.LeakyReLU()
        out2 = m(x)
        out_ref = ref_leaky_relu(self.x_np)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)

        out1 = F.leaky_relu(x, 0.6)
        m = paddle.nn.LeakyReLU(0.6)
        out2 = m(x)
        out_ref = ref_leaky_relu(self.x_np, 0.6)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', [10, 12])
            out = fluid.layers.leaky_relu(x, 0.01)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_leaky_relu(self.x_np)
        self.assertEqual(np.allclose(out_ref, res[0]), True)


if __name__ == '__main__':
    unittest.main()
