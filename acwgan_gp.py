# -*- coding: utf-8 -*-
import time

import tensorboard
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard
from utils import *
from models.libs.resnet_ops import *
from models.libs.ops import linear, gan_batch_norm


class ACWGAN_GP(object):
    model_name = "ACWGAN_GP"  # name for checkpoint

    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir,
                 dim_G=128, dim_D=128, learning_rate=None):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_name = self.model_name
        self.dim_G = dim_G
        self.dim_D = dim_D

        if dataset_name == 'mnist':
            # parameters
            self.input_height = 28
            self.input_width = 28
            self.output_height = 28
            self.output_width = 28
            self.n_iters = 100000

            self.z_dim = z_dim  # dimension of noise-vector
            self.y_dim = 10
            self.c_dim = 1

            # WGAN_GP parameter
            self.lambd = 10 # The higher value, the more stable, but the slower convergence
            self.disc_iters = 5  # The number of critic iterations for one-step of generator

            # train
            self.learning_rate = 0.0002 if learning_rate is None else learning_rate
            self.beta1 = 0.0
            self.beta2 = 0.9

            # test
            self.sample_num = 64  # number of generated images to be saved

            # code
            self.len_discrete_code = 10  # categorical distribution (i.e. label)
            self.len_continuous_code = 2  # gaussian distribution (e.g. rotation, thickness)

            # load mnist
            self.data_X, self.data_y = load_mnist(self.dataset_name)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // (self.batch_size * self.disc_iters)

        else:
            raise NotImplementedError


    def discriminator(self, x, update_collection, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            output = tf.reshape(x, [-1, self.output_height, self.output_width, self.c_dim])
            if self.dataset_name in ('mnist'):
                output = ResidualBlockDisc(output, self.dim_D, spectral_normed=False, update_collection=update_collection, name="d_residual_block")
                output = ResidualBlock(output, None, self.dim_D, 3, resample='down', spectral_normed=False,
                                       update_collection=update_collection, name='d_res1')
                output = ResidualBlock(output, None, self.dim_D, 3, resample=None, spectral_normed=False,
                                       update_collection=update_collection, name='d_res2')
                output = ResidualBlock(output, None, self.dim_D, 3, resample=None, spectral_normed=False,
                                       update_collection=update_collection, name='d_res3')
                output = tf.nn.relu(output)
                output = tf.reduce_mean(output, axis=[1,2]) # global sum pooling
                output_logits = linear(output, 1, spectral_normed=False, update_collection=update_collection, name='d_output')
                output_acgan = linear(output, self.y_dim, spectral_normed=False, update_collection=update_collection,
                                      name='d_acgan_output')
            else:
                raise NotImplementedError("do not support dataset {}".format(self.dataset_name))

            return output_logits, output_acgan

    def generator(self, z, y, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            onehot = tf.one_hot(y, depth=self.y_dim, dtype=tf.float32)
            z = tf.concat([z, onehot], axis=1)
            if self.dataset_name == "mnist":
                output = linear(z, 7 * 7 * self.dim_G, name='g_fc1')
                output = tf.reshape(output, [-1, 7, 7, self.dim_G])
                output = ResidualBlock(output, y, self.dim_G, 3, resample='up', name='g_res1', n_labels=self.y_dim)
                output = ResidualBlock(output, y, self.dim_G, 3, resample='up', name='g_res2', n_labels=self.y_dim)
                output = gan_batch_norm(output, name='g_out')
                output = tf.nn.relu(output)
                output = conv2d(output, 1, 3, he_init=False, name='g_final')
                output = tf.sigmoid(output)

            return output

    def build_model(self):
        # some parameters
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # 图片
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        self.inputs += tf.random_uniform(shape=self.inputs.get_shape().as_list(), minval=0., maxval=1/255.) # dequantize

        # 噪声
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')
        self.y = tf.placeholder(tf.int32, [bs], name='y')
        self.lr = tf.placeholder(tf.float32, (), name='lr')

        """ Loss Function """

        # 生成器假图像输出
        D_real_logits, code_real_logits = self.discriminator(self.inputs, reuse=False, update_collection='spectral_norm')

        # 分类器假图像输出
        G = self.generator(self.z, self.y, reuse=False)
        D_fake_logits, code_fake_logits = self.discriminator(G, reuse=True, update_collection='NO_OPS')

        # 计算分类器损失函数
        d_loss_real = - tf.reduce_mean(D_real_logits)
        d_loss_fake = tf.reduce_mean(D_fake_logits)
        acgan_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=code_real_logits, labels=self.y))
        acgan_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=code_fake_logits, labels=self.y))
        self.acgan_loss_real = acgan_loss_real
        self.acgan_loss_fake = acgan_loss_fake
        self.acgan_real_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.cast(tf.argmax(code_real_logits, axis=1), tf.int32), self.y)))
        self.acgan_fake_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.cast(tf.argmax(code_fake_logits, axis=1), tf.int32), self.y)))

        self.d_loss = d_loss_real + d_loss_fake + acgan_loss_real

        # 计算生成器损失函数
        self.g_loss = - d_loss_fake + acgan_loss_fake

        self.update_op = tf.group(*tf.get_collection("spectral_norm"))

        """ 梯度惩罚 """
        alpha = tf.random_uniform(shape=self.inputs.get_shape(), minval=0., maxval=1.)
        differences = G - self.inputs
        interpolates = self.inputs + (alpha * differences)#既不是完全真实，也不是完全生成
        D_inter = self.discriminator(interpolates, reuse=True, update_collection='NO_OPS')[0]
        gradients = tf.gradients(D_inter, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        self.d_loss += self.lambd * gradient_penalty

        """ 训练 """
        # divide trainable variables into a group for D and a group for G
        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        # 优化
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1) \
            .minimize(self.d_loss, var_list=d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=self.beta1) \
            .minimize(self.g_loss, var_list=g_vars)

        """" 测试 """
        # 生成假图像
        self.fake_images = self.generator(self.z, self.y, reuse=True)

        """ 汇总 """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)


        # 最终汇总
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])

    def train(self):
        #  初始化所有变量
        tf.global_variables_initializer().run()

        #  图输入可视化训练结果
        self.sample_z = np.random.normal(size=(self.batch_size, self.z_dim))
        self.test_codes = np.argmax(self.data_y[0:self.batch_size], axis=1)

        #  保存模型的保护程序
        self.saver = tf.train.Saver()

        #  累加计数器
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)


        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 0
            print(" [!] Load failed...")

        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            # 获取批次数据
            random_state = np.random.get_state()
            np.random.shuffle(self.data_X)
            np.random.set_state(random_state)
            np.random.shuffle(self.data_y)

            for idx in range(start_batch_id, self.num_batches):
                decay = np.maximum(0.0, 1. - counter / (self.n_iters - 1)) if self.dataset_name != 'celebA' else 1.
                batch_images = self.data_X[idx * self.batch_size * self.disc_iters :(idx + 1) * self.batch_size * self.disc_iters]
                if self.dataset_name in ('svhn', 'celebA'):
                    batch_images = (batch_images - 0.5) * 2.
                batch_y = np.argmax(self.data_y[idx * self.batch_size * self.disc_iters : (idx + 1) * self.batch_size * self.disc_iters], axis=1)
                for i in range(self.disc_iters):
                    this_input = batch_images[i * self.batch_size: (i+1) * self.batch_size]
                    this_y = batch_y[i * self.batch_size: (i+1) * self.batch_size]
                    batch_z = np.random.normal(size=[self.batch_size, self.z_dim]).astype(np.float32)
                    _, summary_str, d_loss, real_acc, fake_acc, acgan_l1, acgan_l2 = self.sess.run([self.d_optim, self.d_sum, self.d_loss,
                                                            self.acgan_real_acc, self.acgan_fake_acc, self.acgan_loss_real, self.acgan_loss_fake],
                                                       feed_dict={self.inputs: this_input,
                                                                  self.z: batch_z,
                                                                  self.y: this_y,
                                                                  self.lr: self.learning_rate * decay})
                    self.writer.add_summary(summary_str, counter)

                batch_z = np.random.normal(size=[self.batch_size, self.z_dim]).astype(np.float32)
                random_y = np.random.choice(self.y_dim, self.batch_size).astype(np.int32)


                _, summary_str_g, g_loss, acgan_l2 = self.sess.run(
                    [self.g_optim, self.g_sum, self.g_loss, self.acgan_loss_fake],
                    feed_dict={self.z: batch_z,
                               self.y: random_y,
                               self.lr: self.learning_rate * decay})
                self.writer.add_summary(summary_str_g, counter)

                counter += 1

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))
                print("ACGAN accuracy -- real: {}, fake: {}".format(real_acc, fake_acc))

                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z, self.y: self.test_codes})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(
                                    self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))

                if counter == self.n_iters:
                    self.save(self.checkpoint_dir, counter)
                    self.visualize_results(epoch)
                    return

            start_batch_id = 0

            self.save(self.checkpoint_dir, counter)

            self.visualize_results(epoch)

        self.save(self.checkpoint_dir, counter)

    def visualize_results(self, epoch):
        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        z_sample = np.random.normal(size=(self.batch_size, self.z_dim))

        """ random noise, random discrete code, fixed continuous code """
        y = np.random.choice(self.len_discrete_code, self.batch_size)

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y})

        save_images(samples[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

        """ specified condition, random noise """
        n_styles = 10  # must be less than or equal to self.batch_size

        np.random.seed()
        si = np.random.choice(self.batch_size, n_styles)

        for l in range(self.len_discrete_code):
            y = np.zeros(self.batch_size, dtype=np.int64) + l

            samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.y: y})
            save_images(samples[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim],
                        check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_class_%d.png' % l)

            samples = samples[si, :, :, :]

            if l == 0:
                all_samples = samples
            else:
                all_samples = np.concatenate((all_samples, samples), axis=0)

        """ save merged images to check style-consistency """
        canvas = np.zeros_like(all_samples)
        for s in range(n_styles):
            for c in range(self.len_discrete_code):
                canvas[s * self.len_discrete_code + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

        save_images(canvas, [n_styles, self.len_discrete_code],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
