import tensorflow as tf
import vgg
import numpy as np
import cv2
from skimage import io
import os

# -----------------------------------------准备数据--------------------------------------
#这里以单张图片作为示例，简单说明原理
image = cv2.imread('./cat.18.jpg')
print(image.shape)
res_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)#vgg_16有全连接层，需要固定输入尺寸
print(res_image.shape)
res_image = np.expand_dims(res_image, axis=0)#网络输入为四维[batch_size, height, width, channels]
print(res_image.shape)
labels = [[1,0]]#标签

# -----------------------------------------恢复图------------------------------------------
#恢复图的方式有很多，这里采用手动构造一个跟保存权重时一样的graph
graph = tf.get_default_graph()

input = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='inputs')
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='labels')

# net=[batch, 2]其中2表示二分类，注意官网给出的vgg_16最终的输出没有经过softmax层
net, end_points = vgg.vgg_16(input, num_classes=2)  # 保存的权重模型针对的num_classes=1000，这里改为num_classes=2，因此最后一层需要重新训练
print(net, end_points)  # net是网络的输出；end_points是所有变量的集合

#add more operations to the graph
y = tf.nn.softmax(net)  # 输出0-1之间的概率
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='vgg_16/fc8')  # 注意这里的scope是定义graph时 name_scope的名字，不要加:0
print(output_vars)

# loss只作用在var_list列表中的变量，也就是说只训练var_list中的变量，其余变量保持不变。若不指定var_list，则默认重新训练所有变量
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy,var_list=output_vars)

# ----------------------------------------恢复权重------------------------------------------
var = tf.global_variables()  # 获取所有变量
print(var)
# var_to_restore = [val for val in var if 'conv1' in val.name or 'conv2' in val.name]#保留变量中含有conv1、conv2的变量
var_to_restore = [val for val in var if 'fc8' not in val.name]  # 保留变量名中不含有fc8的变量
print(var_to_restore)

saver = tf.train.Saver(var_to_restore)  # 恢复var_to_restore列表中的变量（最后一层变量fc8不恢复）

with tf.Session() as sess:
	# restore恢复变量值也是变量初始化的一种方式，对于没有restore的变量需要单独初始化
	# 注意如果使用全局初始化，则应在全局初始化后再调用saver.restore()。相当于先通过全局初始化赋值，再通过restore重新赋值。
	saver.restore(sess, './vgg_16.ckpt')  # 权重保存为.ckpt则需要加上后缀

	var_to_init = [val for val in var if 'fc8' in val.name]  # 保留变量名中含有fc8的变量

	# tf.variable_initializers(tf.global_variables())等价于tf.global_variables_initializer()
	sess.run(tf.variables_initializer(var_to_init))  # 没有restore的变量需要单独初始化
	# sess.run(tf.global_variables_initializer())

	# 用w1，w8测试权重恢复成功没有.正确的情况应该是：w1的值不变，w8的值随机
	w1 = sess.graph.get_tensor_by_name('vgg_16/conv1/conv1_1/weights:0')
	print(sess.run(w1, feed_dict={input: res_image}))

	w8 = sess.graph.get_tensor_by_name('vgg_16/fc8/weights:0')
	print('w8', sess.run(w8, feed_dict={input: res_image}))
	
	sess.run(train_op, feed_dict={input:res_image, y_:labels})





	


