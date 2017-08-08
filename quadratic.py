# 写経: http://qiita.com/elm200/items/98777cc184437a1f82c6
# python quadratic.py

import numpy as np
import chainer


class Quadratic(chainer.Link):
    def __init__(self):
        super().__init__(
            x=(1)
        )
        self.x.data = np.array([100], dtype=np.float32)

    def forward(self):
        # y = (x - 1)^2 + 3
        # x = 1 のとき y が最小
        return self.x ** 2 - 2 * self.x + 4


model = Quadratic()
# SGD: Stochastic (確率的) Gradient Descent
# サンプル1つずつとってそれについて最適化していくやつ
# lr = 学習係数
optimizer = chainer.optimizers.SGD(lr=0.1)
optimizer.use_cleargrads()
optimizer.setup(model)

for i in range(80):
    model.cleargrads()
    # x の関数を計算して y をだす
    y = model.forward()
    # 傾きを計算 (?) これで model.x.grad が計算される？
    # y ってオブジェクトなのか
    y.backward()
    print("=== Epoch %d ===" % (i + 1))
    print("model.x.data = %f" % model.x.data)
    print("y.data = %f" % y.data)
    print("model.x.grad = %f" % model.x.grad)
    print()
    optimizer.update()
