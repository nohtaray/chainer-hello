import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


print("--- Forward / Backward Computation ---")
print()

x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)

# Define-by-Run のしくみ。y を計算すること自体がネットワークの定義になる
# y = x^2 - 2x + 1
y = x**2 - 2 * x + 1

print("x.data = %f" % x.data)
print()
print("y = x^2 - 2x + 1")
print("y.data = %f" % y.data)

# y も Variable。計算の過程が保持されてる。
# error backpropagation を走らせる。
y.backward()

# これで x.grad にアクセスしたら x の勾配がとれる
print()
print("x.grad = %f" % x.grad)

print("--")
# 複数の x について計算
x_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
x = Variable(x_data)

y = x**2 - 2 * x + 1
print("x.data =")
print(x.data)
print()
print("y = x^2 - 2x + 1")
print()
print("y.data =")
print(y.data)
print()

# x が複数の場合 initial error を指定してあげないといけない (?)
y.grad = np.ones((2, 3), dtype=np.float32)
y.backward()

print("x.grad =")
print(x.grad)
