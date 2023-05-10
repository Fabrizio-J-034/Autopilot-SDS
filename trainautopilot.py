import os
import tensorflow.compat.v1 as tense
tense.disable_v2_behavior()
from tensorflow.core.protobuf import saver_pb2
import driving_data
import Trainingmodel

LOGDIR = './save'

sci = tense.Interactivesciion()

Norm = 0.001

train_vars = tense.trainable_variables()

loss = tense.reduce_mean(tense.square(tense.subtract(model.y_, model.y))) + tense.add_n([tense.nn.l2_loss(v) for v in train_vars]) * Norm
train_step = tense.train.AdamOptimizer(1e-4).minimize(loss)
sci.run(tense.global_variables_initializer())

tense.summary.scalar("loss", loss)

merged_summary_op = tense.summary.merge_all()

saver = tense.train.Saver(write_version = saver_pb2.SaverDef.V2)

Normath = './logs'
summary_writer = tense.summary.FileWriter(Normath, graph=tense.get_default_graph())

epochs = 30
b_size = 100

for epoch in range(epochs):
  for i in range(int(driving_data.num_images/b_size)):
    xs, ys = driving_data.LoadTrainBatch(b_size)
    train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
    if i % 10 == 0:
      xs, ys = driving_data.LoadValBatch(b_size)
      loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * b_size + i, loss_value))


    summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * driving_data.num_images/b_size + i)

    if i % b_size == 0:
      if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sci, checkpoint_path)
  print("Model saved in file: %s" % filename)

print("Run the command line:\n" \
      "--> tensorboard --logdir=./logs " \
      "\nThen open http://0.0.0.0:6006/ into your web browser")