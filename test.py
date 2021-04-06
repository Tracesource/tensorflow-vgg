import tensorflow as tf
import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 只显示 warning 和 Error 
hello = tf.constant('Hello, TensorFlow!')  #初始化一个TensorFlow的常量
sess = tf.compat.v1.Session()  #启动一个会话
print(sess.run(hello))  