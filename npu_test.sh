#!/bin/bash

#python3 -m pip uninstall paddlepaddle -y
#python3 -m pip install python/dist/paddlepaddle-0.0.0-cp37-cp37m-linux_x86_64.whl
pip uninstall paddlepaddle -y
pip install -U python/dist/paddlepaddle-0.0.0-cp37-cp37m-linux_aarch64.whl
test_cases=$(ctest -N -V | grep "op_npu$" )
while read -r line; do
    if [[ "$line" == "" ]]; then
        continue
    fi
    read testcase <<< $(echo "$line"|grep -oEi "\w+$")
    if [[ "$single_card_tests" == "" ]]; then
        single_card_tests="^$testcase$"
    else
        single_card_tests="$single_card_tests|^$testcase$"
        ctest -R $testcase --output-on-failure -E "(^test_softmax_with_cross_entropy_op_npu$|^test_slice_op_npu$|^test_logical_op_npu$|^test_layer_norm_op_npu$)"
    fi
done <<< "$test_cases";

