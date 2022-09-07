import tensorflow as tf
import numpy as np

class TF_utils:
	def infer_shape(self, x):
		x = tf.convert_to_tensor(x)

		# If unknown rank, return dynamic shape
		if x.shape.dims is None:
			return tf.shape(x)

		static_shape = x.shape.as_list()
		dynamic_shape = tf.shape(x)

		ret = []
		for i in range(len(static_shape)):
			dim = static_shape[i]
			if dim is None:
				dim = dynamic_shape[i]
			ret.append(dim)

		return ret

	def infer_shape_invariants(self, tensor):
		
		shape = tensor.shape.as_list()
		for i in range(1, len(shape) - 1):
			shape[i] = None
		return tf.TensorShape(shape)

	def merge_first_two_dims(self, tensor):
		shape = self.infer_shape(tensor)
		shape[0] *= shape[1]
		shape.pop(1)
		return tf.reshape(tensor, shape)

	def split_first_two_dims(self, tensor, dim_0, dim_1):
		shape = self.infer_shape(tensor)
		new_shape = [dim_0] + [dim_1] + shape[1:]
		return tf.reshape(tensor, new_shape)

	def tile_to_beam_size(self, tensor, beam_size):
		"""
		Tiles a given tensor by beam_size.
		[bz,1] = > [bz, beam_size, 1]
		"""
		tensor = tf.expand_dims(tensor, axis=1)
		tile_dims = [1] * tensor.shape.ndims
		tile_dims[1] = beam_size

		return tf.tile(tensor, tile_dims)

	def tile_batch(self, tensor, batch_size):
		#
		shape = self.infer_shape(tensor)
		tile_dims = [1] * (tensor.shape.ndims + 1)
		tile_dims[1] = batch_size

		tensor = tf.tile(tf.expand_dims(self, tensor, axis=1), tile_dims)
		shape[0] = shape[0] * batch_size

		return tf.reshape(tensor, shape)

	def gather_2d(self, params, indices, name=None):
		""" Gather the 2nd dimension given indices
		:param params: A tensor with shape [batch_size, M, ...]
		:param indices: A tensor with shape [batch_size, N]
		:param name: An optional string
		:return: A tensor with shape [batch_size, N, ...]
		"""
		batch_size = tf.shape(params)[0]  #
		range_size = tf.shape(indices)[1]
		batch_pos = tf.range(batch_size * range_size) // range_size
		batch_pos = tf.reshape(batch_pos, [batch_size, range_size])
		indices = tf.stack([batch_pos, indices], axis=-1)
		output = tf.gather_nd(params, indices, name=name)

		return output

	def split_input(self, xs, ys, gpu_nums):
		"""
		split input
		:param xs: articles
		:param ys: summaries
		:param gpu_nums: gpu numbers
		:return: split input by gpu numbers
		"""
		xs = [tf.split(x, num_or_size_splits=gpu_nums, axis=0) for x in xs]
		ys = [tf.split(y, num_or_size_splits=gpu_nums, axis=0) for y in ys]
		feature_num = len(xs)
		return [tuple([xs[j][i] for j in range(feature_num)]) for i in range(gpu_nums)], [(ys[0][i], ys[1][i], ys[2][i])
																						  for i in range(gpu_nums)]

	def np_tile_to_beam_size(self, vec, beam_size):
		vec = np.expand_dims(vec, axis=1)
		return np.tile(vec, [1, beam_size, 1])

	def np_merge_first_two_dims(self, vec):
		shape = vec.shape
		return vec.reshape((shape[0] * shape[1], -1))

	def convert_idx_to_token_tensor(self, inputs, idx2token):
		'''Converts int32 tensor to string tensor.
		inputs: 1d int32 tensor. indices.
		idx2token: dictionary

		Returns
		1d string tensor.
		'''

		def my_func(inputs):
			return " ".join(idx2token[elem] for elem in inputs)

		return tf.py_func(my_func, [inputs], tf.string)
