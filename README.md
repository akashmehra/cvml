ml-experiments
==============
Starting it by adding the mnist loader written in python based on the data available on [Prof. Yann LeCun's page](http://yann.lecun.com/exdb/mnist/)

<b>Usage</b>:

Quickly test in ipython or python shell, do the following (till the time I provide an install script):

```python
import sys
sys.append(<path to python script>)

import mnist
trainset = mnist.traindataset()
X_train = trainset.X
Y_train = trainset.Y

testset = mnist.testdataset()
X_test = testset.X
Y_test = testset.Y
