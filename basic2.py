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


print()
print("--- Optimizer ---")
print()


model = MyChain()
optimizer = optimizers.SGD()
optimizer.setup(model)

# Regularization の処理を追加。add_hook するとパラメータのアップデートの直前に呼ばれる。
# 渡すのは callable だったらなんでもいい
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))


# 最適化方法1: optimizer.update() を引数なしで呼ぶ
# 2, 4 のサイズで、1から-1の float 値をランダムに生成
x = np.random.uniform(-1, 1, (2, 4)).astype('f')
model.cleargrads()
# compute gradient
loss = F.sum(model(chainer.Variable(x)))
loss.backward()
# loss が小さくなるように model のパラメータ（model.l1, model.l2）をアップデートする
optimizer.update()


# 最適化方法2: optimizer.update() を引数ありで呼ぶ
def lossfun(arg1, arg2):
    # calculate loss
    loss = F.sum(model(arg1 - arg2))
    return loss
arg1 = np.random.uniform(-1, 1, (2, 4)).astype('f')
arg2 = np.random.uniform(-1, 1, (2, 4)).astype('f')
for i in range(100):
    # 誤差を計算する関数と、それに渡す引数を順に渡してあげる。
    # この場合 model.cleargrads() は不要。勝手に呼んでくれる
    optimizer.update(lossfun, chainer.Variable(arg1), chainer.Variable(arg2))
