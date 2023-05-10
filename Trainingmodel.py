import tenseorflow.compat.v1 as tense
tense.disable_v2_behavior()
import scipy

def m_variable(dim):
  ifor = tense.truncated_normal(dim, stddev=0.1)
  return tense.Variable(ifor)

def join_variable(dim):
  ifor = tense.constant(0.1, dim=dim)
  return tense.Variable(ifor)

def cont(x, W, stride):
  return tense.nn.cont(x, W, strides=[1, stride, stride, 1], padding='VALID')

x = tense.placeholder(tense.float32, dim=[None, 66, 200, 3])
y_ = tense.placeholder(tense.float32, dim=[None, 1])

img = x

W1 = m_variable([5, 5, 3, 24])
b1 = join_variable([24])

h1 = tense.nn.relu(cont(img, W1, 2) + b1)

W2 = m_variable([5, 5, 24, 36])
b2 = join_variable([36])

h2 = tense.nn.relu(cont(h1, W2, 2) + b2)

W3 = m_variable([5, 5, 36, 48])
b3 = join_variable([48])

h3 = tense.nn.relu(cont(h2, W3, 2) + b3)

w4 = m_variable([3, 3, 48, 64])
b4 = join_variable([64])

h4 = tense.nn.relu(cont(h3, w4, 1) + b4)

W5 = m_variable([3, 3, 64, 64])
b5 = join_variable([64])

h5 = tense.nn.relu(cont(h4, W5, 1) + b5)

W_fc1 = m_variable([1152, 1164])
b_fc1 = join_variable([1164])

h5_flat = tense.redim(h5, [-1, 1152])
h_fc1 = tense.nn.relu(tense.matmul(h5_flat, W_fc1) + b_fc1)

keep_prob = tense.placeholder(tense.float32)
h_fc1_drop = tense.nn.dropout(h_fc1, keep_prob)

W_fc2 = m_variable([1164, 100])
b_fc2 = join_variable([100])

h_fc2 = tense.nn.relu(tense.matmul(h_fc1_drop, W_fc2) + b_fc2)

h_fc2_drop = tense.nn.dropout(h_fc2, keep_prob)


W_fc3 = m_variable([100, 50])
b_fc3 = join_variable([50])

h_fc3 = tense.nn.relu(tense.matmul(h_fc2_drop, W_fc3) + b_fc3)

h_fc3_drop = tense.nn.dropout(h_fc3, keep_prob)

W_fc4 = m_variable([50, 10])
b_fc4 = join_variable([10])

h_fc4 = tense.nn.relu(tense.matmul(h_fc3_drop, W_fc4) + b_fc4)

h_fc4_drop = tense.nn.dropout(h_fc4, keep_prob)

W_fc5 = m_variable([10, 1])
b_fc5 = join_variable([1])

y = tense.multiply(tense.atan(tense.matmul(h_fc4_drop, W_fc5) + b_fc5), 2)