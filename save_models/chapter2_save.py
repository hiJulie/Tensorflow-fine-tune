import tensorflow as tf
from six.moves import xrange
import os

w1 = tf.Variable(tf.random_normal(shape=[2]), name='w11')#变量w1在内存中的名字是w11；恢复变量时应该与name的名字保持一致
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w22')
w3 = tf.Variable(tf.random_normal(shape=[5]), name='w33')

#保存一部分变量[w1,w2];只保存最近的5个模型文件;每2小时保存一次模型
saver = tf.train.Saver([w1, w2],max_to_keep=5, keep_checkpoint_every_n_hours=2)
save_path = './checkpoint_dir/MyModel'#定义模型保存的路径./checkpoint_dir/及模型名称MyModel

# Launch the graph and train, saving the model every 1,000 steps.
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for step in xrange(100):
		if step % 10 == 0:
			# 每隔step=10步保存一次模型（ keep_checkpoint_every_n_hours与global_step可同时使用，表示'与'，通常任选一个就够了）；
			#每次会在保存的模型名称后面加上global_step的值作为后缀
			# write_meta_graph=False表示不保存图
			saver.save(sess, save_path, global_step=step, write_meta_graph=False)
			# 如果模型文件中没有保存网络图，则使用如下语句保存一张网络图（由于网络图不变，只保存一次就行）
			if not os.path.exists('./checkpoint_dir/MyModel.meta'):
				# saver.export_meta_graph(filename=None, collection_list=None,as_text=False,export_scope=None,clear_devices=False)
				# saver.export_meta_graph()仅仅保存网络图；参数filename表示网络图保存的路径即网络图名称
				saver.export_meta_graph('./checkpoint_dir/MyModel.meta')#定义网络图保存的路径./checkpoint_dir/及网络图名称MyModel.meta
                                #注意：tf.train.export_meta_graph()等价于tf.train.Saver.export_meta_graph()