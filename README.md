cvml
==============
Starting it by adding the mnist loader written in python based on the data available on [Prof. Yann LeCun's page](http://yann.lecun.com/exdb/mnist/)

Variables and code style are exactly similar to the example [here](https://github.com/andresy/mnist) in lua (using torch7).

<b>Usage</b>:

Quickly test in ipython or python shell, do the following (till the time I provide an install script):

```python
import sys
sys.append(<path to python script>)

import mnist
trainset = mnist.traindataset()
X_train = trainset.X
Y_train = trainset.Y
```
i-th training example and the label is retrieved as follows:
```python
ex_i = X_train[i] # gives a (28,28) numpy array
label_i = Y_train[i] # label [0-9]
```
test set can be loaded using testdataset function.

```python
testset = mnist.testdataset()
X_test = testset.X
Y_test = testset.Y
```

