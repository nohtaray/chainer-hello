# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


print("--- Write a model as a chain ---")
print()

l1 = L.Linear(4, 3)
l2 = L.Linear(3, 2)


# ユニット数が 4 -> 3 -> 2 のネットワーク
def my_forward(x):
    h = l1(x)
    return l2(h)


x = Variable(np.array([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=np.float32))
y = my_forward(x)

y.grad = np.ones((2, 2), dtype=np.float32)
x.backward()

print("x.data = ")
print(x.data)
print()

print("l1.W = ")
print(l1.W.data)
print("l1.b = ")
print(l1.b.data)
print()
print("l2.W = ")
print(l2.W.data)
print("l2.b = ")
print(l2.b.data)
print()

print("y = (x * l1.W^T + l1.b) * l2.W^T + l2.b")
print()

print("y.data = ")
print(y.data)


print()
print("--")
print()


# Chain クラスを利用
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(4, 3)
            self.l2 = L.Linear(3, 2)

    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)


my_chain = MyChain()
x = Variable(np.array([[1, 2, 3, 4], [4, 5, 6, 7], [6, 7, 8, 9]], dtype=np.float32))
y = my_chain(x)

y.grad = np.ones((3, 2), dtype=np.float32)
x.backward()

print("x.data = ")
print(x.data)
print()

print("y = (x * l1.W^T + l1.b) * l2.W^T + l2.b")
print()

print("y.data = ")
print(y.data)


# 他に ChainList というクラスもあって、動的に層の数を変えられる。
# 層の数が決まってるんだったら Chain クラスを使うのを推奨。
