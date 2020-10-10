# Brief about tensors:

# Its an nd-array
# Has GPU support
# Computational graph / Backpropagation
# immutable

import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Rank 1 Tensor
x = tf.constant([1,2,3])

# Rank 2
x = tf.constant([[1,2,3], [4,5,6]])

# Rank 3
x = tf.ones((3,3))
x = tf.zeros((3,3))
x = tf.eye(3) # Diagonals are filled with "1" and others are "0"

# Random tensor
x = tf.random.normal((3,3), mean=0, stddev=1)

# Uniform distribution
x = tf.random.uniform((3,3), minval=0, maxval=1) # there the values are uniformally distributed

# Range tensor
x = tf.range(10)

# Cast tensor (cast datatype for example)
x = tf.cast(x, dtype=tf.float32)

# Opeartions on tensor
x = tf.constant([1,2,3])
y = tf.constant([4,5,6])

# add
z = tf.add(x,y)

# Another way of adding
z = x+y

# Subtraction
z = x-y

# divide
z = x/y

# Multiplication
z = x*y

# Computing dot product (does product of x and y and then add the resultant values)
z = tf.tensordot(x,y, axes=1)

# Element wise exponential product (square root eg.)
z = x ** 2

# Matrix multiplication (number of col in x should match rows in y: simple matrix rule for multiplication)
x = tf.random.normal((2,3))
y = tf.random.normal((3,4))

z = tf.matmul(x,y)

# another way
z = x @ y

# Slicing , indexing (same as numpy or list)
x = tf.constant([[1,2,3,4], [5,6,7,8]]) # 2-d array (2d tensor)

# access only the first row
# print(x[0])

# access all the rows in column 0
# print (x[:, 0])

# Access only row 0 and all columns
# print (x[0, :])

# Access ROW 0 AND COL 1
# print(x[0,1])

# original
x = tf.random.normal((2,3))
# print(x)
# reshaping
x = tf.reshape(x, (3,2)) # for reshaping we need to maintain the original shape order.
# print(x)
x = tf.reshape(x, (6)) # as original (2,3) -> 6 values
# print(x)

# Convert x to numpy array
x = x.numpy()
print(x)
print(type(x)) # <class 'numpy.ndarray'>

# convert back to tensor
x = tf.convert_to_tensor(x)
print(type(x)) # <class 'tensorflow.python.framework.ops.EagerTensor'> this tensor is a eager tensor

# String tensor
x = tf.constant("chandan")
print(x)

# Multi string tensor
x = tf.constant(["chandan", "sanjana"])
print(x)

# Constant tensorflows are immutable, but variable tensors can be changed
x = tf.Variable([1,2,3])
print(x) # <tf.Variable 'Variable:0' shape=(3,) dtype=int32, numpy=array([1, 2, 3], dtype=int32)>