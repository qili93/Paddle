import paddle
import numpy as np

paddle.set_device("gpu")
# paddle.set_device("cpu")

def test_sum_16(temp):
  x = paddle.tile(temp, repeat_times=[1, 16, 1])
  out = paddle.sum(x, axis=-1)
  print("out =", out)

def test_sum_32(temp):
  x = paddle.tile(temp, repeat_times=[1, 32, 1])
  out = paddle.sum(x, axis=-1)
  print("out =", out)

def test_sum_256(temp):
  x = paddle.tile(temp, repeat_times=[1, 256, 1])
  out = paddle.sum(x, axis=-1)
  print("out =", out)


if __name__ == "__main__":
  temp = paddle.reshape(paddle.to_tensor([1.0, 0.0, 0.0, 0.0]), [1, 1, 4])
  test_sum_16(temp)
