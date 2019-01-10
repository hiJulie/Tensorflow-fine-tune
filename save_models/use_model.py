import tensorflow as tf
import vgg
import numpy as np
import cv2
from skimage import io
import os

#-----------------------------------------准备数据--------------------------------------
image = cv2.imread('./cat.18.jpg')
print(image.shape)
res_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
print(res_image.shape)
res_image = np.expand_dims(res_image, axis=0)
print(res_image.shape)

#-----------------------------------------恢复图------------------------------------------
graph = tf.Graph
input = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='inputs')
net, end_points = vgg.vgg_16(input, num_classes=1000)
print(end_points)#获取tensor的名字，便于后续通过get_tensor_by_name()获取指定tensor

saver = tf.train.Saver()

#----------------------------------------恢复权重------------------------------------------
with tf.Session() as sess:
	saver.restore(sess, './vgg_16.ckpt')#权重保存为.ckpt则需要加上后缀
	
	"""
	恢复出来的模型有四种用途：
	1.查看模型参数
	2.直接使用原始模型进行测试
	3.扩展原始模型（直接使用扩展后的网络进行测试，扩展后需要重新训练的情况见微调部分）
	4.微调：使用先前训练好的权重参数进行初始化，在此基础上对网络的全部或者局部参数进行重新训练
	"""
	
#----------------------------------------1.查看模型参数---------------------------------------
	"""
	   查看恢复的模型参数
	   tf.trainable_variables()查看的是所有可训练的变量；
	   tf.global_variables()获得的与tf.trainable_variables()类似，只是多了一些非trainable的变量，比如定义时指定为trainable=False的变量；
	   sess.graph.get_operations()则可以获得几乎所有的operations相关的tensor
	   """
	tvs = [v for v in tf.trainable_variables()]
	print('获得所有可训练变量的权重:')
	for v in tvs:
		print(v.name)
		print(sess.run(v))

	gv = [v for v in tf.global_variables()]
	print('获得所有变量:')
	for v in gv:
		print(v.name, '\n')

	# sess.graph.get_operations()可以换为tf.get_default_graph().get_operations()
	ops = [o for o in sess.graph.get_operations()]
	print('获得所有operations相关的tensor:')
	for o in ops:
		print(o.name, '\n')
	
#--------------------------------------2.直接使用原始模型进行测试-----------------------------
	# Get input and output tensors
	# 需要特别注意，get_tensor_by_name后面传入的参数，如果没有重复，需要在后面加上“:0”
	# sess.graph等价于tf.get_default_graph()
	input = sess.graph.get_tensor_by_name('inputs:0')
	output = sess.graph.get_tensor_by_name('vgg_16/fc8/squeezed:0')
	
	# Run forward pass to calculate pred
	# 使用不同的数据运行相同的网络，只需将新数据通过feed_dict传递到网络即可。
	pred = sess.run(output, feed_dict={input: res_image})
	# 得到使用vgg网络对输入图片的分类结果
	print(np.argmax(pred, 1))
	
# -------------------------------------3.扩展原始模型-----------------------------
	#明确的网络的输入输出，通过get_tensor_by_name()获取变量
	input = sess.graph.get_tensor_by_name('inputs:0')
	output = sess.graph.get_tensor_by_name('vgg_16/fc8/squeezed:0')
	
	# add more operations to the graph
	#这里只是简单示例，也可以加上新的网络层。
	pred = tf.argmax(output, 1)

	#使用不同的数据运行相同的网络，只需将新数据通过feed_dict传递到网络即可。
	pred = sess.run(pred, feed_dict={input:res_image})
	print(pred)

















	#
	
	# fc7 = sess.graph.get_tensor_by_name('vgg_16/dropout7/dropout/mul:0')
	# fc8 = sess.graph.get_tensor_by_name('vgg_16/fc8/squeezed:0')
	#
	# print(fc7)
	# fc7 = tf.stop_gradient(fc7)
	#
	# fc7_shape = fc7.get_shape().as_list()
	# print(fc7_shape)
	# num_outputs = 2
	# weights = tf.Variable(tf.truncated_normal([fc7_shape[3],num_outputs], stddev=0.05))
	# biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
	# output = tf.matmul(fc7, weights) + biases
	# pred = tf.nn.softmax(output)










	
	
	#
	# # Get input and output tensors
	# # 需要特别注意，get_tensor_by_name后面传入的参数，如果没有重复，需要在后面加上“:0”
	# #sess.graph等价于tf.get_default_graph()
	# input = sess.graph.get_tensor_by_name('inputs:0')
	# output = sess.graph.get_tensor_by_name('vgg_16/fc8/squeezed:0')
	#
	# # Run forward pass to calculate pred
	# pred = sess.run(output, feed_dict={input:res_image})
	# #得到使用vgg网络对输入图片的分类结果
	# print(np.argmax(pred, 1))







