#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle

paddle.enable_static()


class TestSliceApiWithLoDTensorArray(unittest.TestCase):
    def setUp(self):
        self.shape = (3, 4)
        self.data = np.random.random(size=self.shape).astype('float32')
        self.idx = 0
        self.start = 0
        self.end = 2
        self.axis = 1

        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)

    def set_program_and_run(self, main_program, case_num):
        with fluid.program_guard(main_program):
            x = [
                fluid.data(
                    name='x0', shape=self.shape, dtype="float32"), fluid.data(
                        name='x1', shape=self.shape, dtype="float32"),
                fluid.data(
                    name='x2', shape=self.shape, dtype="float32")
            ]

            for each_x in x:
                each_x.stop_gradient = False

            arr = layers.create_array(dtype="float32")
            for i in range(3):
                idx = layers.array_length(arr)
                arr = layers.array_write(x=x[i], i=idx, array=arr)

            if case_num == 1:
                self.sliced_arr = output = arr[0]

            elif case_num == 2:
                end = fluid.layers.array_length(
                    arr) - 1  # dtype of end is int64
                self.sliced_arr = slice_arr = arr[self.start:end]
                output, _ = fluid.layers.tensor_array_to_tensor(
                    slice_arr, axis=self.axis, use_stack=True)
            elif case_num == 3:
                value_int64 = fluid.layers.fill_constant([1], "int64",
                                                         2147483648)
                self.sliced_arr = slice_arr = arr[self.start:value_int64]
                output, _ = fluid.layers.tensor_array_to_tensor(
                    slice_arr, axis=self.axis, use_stack=True)

            loss = fluid.layers.reduce_sum(output)
            fluid.backward.append_backward(loss)
            g_vars = list(
                map(main_program.global_block().var,
                    [each_x.name + "@GRAD" for each_x in x]))
            self.out, self.g_x0, self.g_x1, self.g_x2 = \
                self.exe.run(main_program,
                             feed = {'x0': self.data,
                                     'x1': self.data,
                                     'x2': self.data},
                             fetch_list=[output] + g_vars)

    def test_case_2(self):
        main_program = fluid.Program()
        self.set_program_and_run(main_program, 2)

        self.assertTrue(
            self.sliced_arr.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY)
        self.assertEqual(self.sliced_arr.shape, self.shape)
        self.assertTrue(
            np.array_equal(
                self.out, np.stack(
                    [self.data, self.data], axis=self.axis)))
        self.assertTrue(np.array_equal(self.g_x0, np.ones_like(self.data)))
        self.assertTrue(np.array_equal(self.g_x1, np.ones_like(self.data)))
        self.assertTrue(np.array_equal(self.g_x2, np.zeros_like(self.data)))

    def test_case_3(self):
        main_program = fluid.Program()
        self.set_program_and_run(main_program, 3)

        self.assertTrue(
            self.sliced_arr.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY)
        self.assertEqual(self.sliced_arr.shape, self.shape)
        self.assertTrue(
            np.array_equal(
                self.out,
                np.stack(
                    [self.data, self.data, self.data], axis=self.axis)))
        self.assertTrue(np.array_equal(self.g_x0, np.ones_like(self.data)))
        self.assertTrue(np.array_equal(self.g_x1, np.ones_like(self.data)))
        self.assertTrue(np.array_equal(self.g_x2, np.ones_like(self.data)))


if __name__ == '__main__':
    unittest.main()
