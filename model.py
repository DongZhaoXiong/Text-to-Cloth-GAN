import tensorflow as tf
import ops


class TC_GAN:
    '''
	OPTIONS
	z_dim : Noise dimension 100
	t_dim : Text feature dimension 256
	image_size : Image Dimension 64
	gf_dim : Number of conv in the first layer generator 64
	df_dim : Number of conv in the first layer discriminator 64
	gfc_dim : Dimension of gen untis for for fully connected layer 1024
	caption_vector_length : Caption Vector Length 2400
	batch_size : Batch Size 64
	'''

    def __init__(self, options):
        self.options = options

        self.g_bn0 = ops.batch_norm(name='g_bn0')
        self.g_bn1 = ops.batch_norm(name='g_bn1')
        self.g_bn2 = ops.batch_norm(name='g_bn2')
        self.g_bn3 = ops.batch_norm(name='g_bn3')
        self.g_bn4 = ops.batch_norm(name='g_bn4')
        self.g_bn5 = ops.batch_norm(name='g_bn5')

        self.d_bn1 = ops.batch_norm(name='d_bn1')
        self.d_bn2 = ops.batch_norm(name='d_bn2')
        self.d_bn3 = ops.batch_norm(name='d_bn3')
        self.d_bn4 = ops.batch_norm(name='d_bn4')
        self.d_bn5 = ops.batch_norm(name='d_bn5')
        self.d_bn6 = ops.batch_norm(name='d_bn6')

    def build_model(self):
        img_size = self.options['image_size']
        t_real_image = tf.placeholder('float32', [self.options['batch_size'], img_size, img_size, 3], name='real_image')
        t_wrong_image = tf.placeholder('float32', [self.options['batch_size'], img_size, img_size, 3],
                                       name='wrong_image')
        t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']],
                                        name='real_caption_input')
        t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])

        fake_image = self.generator(t_z, t_real_caption)

        disc_real_image, disc_real_image_logits = self.discriminator(t_real_image, t_real_caption)
        disc_wrong_image, disc_wrong_image_logits = self.discriminator(t_wrong_image, t_real_caption, reuse=False)
        disc_fake_image, disc_fake_image_logits = self.discriminator(fake_image, t_real_caption, reuse=False)

        g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits,
                                                                        labels=tf.ones_like(disc_fake_image)))

        d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_image_logits,
                                                                         labels=tf.ones_like(disc_real_image)))
        d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_wrong_image_logits,
                                                                         labels=tf.zeros_like(disc_wrong_image)))
        d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_image_logits,
                                                                         labels=tf.zeros_like(disc_fake_image)))

        d_loss = d_loss1 + d_loss2 + d_loss3

        t_vars = tf.trainable_variables()
        print('List of all variables')
        for v in t_vars:
            print(v.name)
            print(v)
            self.add_histogram_summary(v.name, v)

        self.add_tb_scalar_summaries(d_loss, g_loss, d_loss1, d_loss2, d_loss3,
                                     )

        self.add_image_summary('Generated Images', fake_image,
                               self.options['batch_size'])
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        input_tensors = {
            't_real_image': t_real_image,
            't_wrong_image': t_wrong_image,
            't_real_caption': t_real_caption,
            't_z': t_z
        }

        variables = {
            'd_vars': d_vars,
            'g_vars': g_vars
        }

        loss = {
            'g_loss': g_loss,
            'd_loss': d_loss
        }

        outputs = {
            'generator': fake_image
        }

        checks = {
            'd_loss1': d_loss1,
            'd_loss2': d_loss2,
            'd_loss3': d_loss3,
            'disc_real_image_logits': disc_real_image_logits,
            'disc_wrong_image_logits': disc_wrong_image,
            'disc_fake_image_logits': disc_fake_image_logits
        }

        return input_tensors, variables, loss, outputs, checks

    def add_tb_scalar_summaries(self, d_loss, g_loss, d_loss1, d_loss2,
                                d_loss3):

        self.add_scalar_summary("D_Loss", d_loss)
        self.add_scalar_summary("G_Loss", g_loss)
        self.add_scalar_summary("D loss-1 [loss for real images]",
                                d_loss1)
        self.add_scalar_summary("D loss-2 [loss for wrong images]",
                                d_loss2)
        self.add_scalar_summary("D loss-3 [loss for fake images]",
                                d_loss3)

    def add_scalar_summary(self, name, var):
        with tf.name_scope('summaries'):
            tf.summary.scalar(name, var)

    def add_histogram_summary(self, name, var):
        with tf.name_scope('summaries'):
            tf.summary.histogram(name, var)

    def add_image_summary(self, name, var, max_outputs=1):
        with tf.name_scope('summaries'):
            tf.summary.image(name, var, max_outputs=max_outputs)

    def build_generator(self):
        img_size = self.options['image_size']
        t_real_caption = tf.placeholder('float32', [self.options['batch_size'], self.options['caption_vector_length']],
                                        name='real_caption_input')
        t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])
        fake_image = self.sampler(t_z, t_real_caption)

        input_tensors = {
            't_real_caption': t_real_caption,
            't_z': t_z
        }

        outputs = {
            'generator': fake_image
        }

        return input_tensors, outputs

    # Sample Images for a text embedding
    def sampler(self, t_z, t_text_embedding):
        tf.get_variable_scope().reuse_variables()

        s = self.options['image_size']
        s2, s4, s8, s16, s32, s64 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(s / 64)

        reduced_text_embedding = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding'))
        z_concat = tf.concat([t_z, reduced_text_embedding], 1)
        z_ = ops.linear(z_concat, self.options['gf_dim'] * 32 * s64 * s64, 'g_h0_lin')
        h0 = tf.reshape(z_, [-1, s64, s64, self.options['gf_dim'] * 32])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = ops.deconv2d(h0, [self.options['batch_size'], s32, s32, self.options['gf_dim'] * 16], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = ops.deconv2d(h1, [self.options['batch_size'], s16, s16, self.options['gf_dim'] * 8], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = ops.deconv2d(h2, [self.options['batch_size'], s8, s8, self.options['gf_dim'] * 4], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = ops.deconv2d(h3, [self.options['batch_size'], s4, s4, self.options['gf_dim'] * 2], name='g_h4')

        h4 = tf.nn.relu(self.g_bn4(h4, train=False))

        h5 = ops.deconv2d(h4, [self.options['batch_size'], s2, s2, self.options['gf_dim'] * 1], name='g_h5')
        h5 = tf.nn.relu(self.g_bn5(h5, train=False))
        h6 = ops.deconv2d(h5, [self.options['batch_size'], s, s, 3], name='g_h6')

        return (tf.tanh(h6) / 2. + 0.5)

    def generator(self, t_z, t_text_embedding):

        s = self.options['image_size']
        s2, s4, s8, s16, s32, s64 = int(s / 2), int(s / 4), int(s / 8), int(s / 16), int(s / 32), int(s / 64)
        reduced_text_embedding = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'], 'g_embedding'))
        z_concat = tf.concat([t_z, reduced_text_embedding], 1)
        z_ = ops.linear(z_concat, self.options['gf_dim'] * 32 * s64 * s64, 'g_h0_lin')
        h0 = tf.reshape(z_, [-1, s64, s64, self.options['gf_dim'] * 32])
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1 = ops.deconv2d(h0, [self.options['batch_size'], s32, s32, self.options['gf_dim'] * 16], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2 = ops.deconv2d(h1, [self.options['batch_size'], s16, s16, self.options['gf_dim'] * 8], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3 = ops.deconv2d(h2, [self.options['batch_size'], s8, s8, self.options['gf_dim'] * 4], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4 = ops.deconv2d(h3, [self.options['batch_size'], s4, s4, self.options['gf_dim'] * 2], name='g_h4')
        h4 = tf.nn.relu(self.g_bn4(h4))

        h5 = ops.deconv2d(h4, [self.options['batch_size'], s2, s2, self.options['gf_dim'] * 1], name='g_h5')
        h5 = tf.nn.relu(self.g_bn5(h5))

        h6 = ops.deconv2d(h5, [self.options['batch_size'], s, s, 3], name='g_h6')

        return (tf.tanh(h6) / 2. + 0.5)

    def discriminator(self, image, t_text_embedding, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = ops.lrelu(ops.conv2d(image, self.options['df_dim'], name='d_h0_conv'))

        h1 = ops.lrelu(self.d_bn1(ops.conv2d(h0, self.options['df_dim'] * 2, name='d_h1_conv')))

        h2 = ops.lrelu(self.d_bn2(ops.conv2d(h1, self.options['df_dim'] * 4, name='d_h2_conv')))

        h3 = ops.lrelu(self.d_bn3(ops.conv2d(h2, self.options['df_dim'] * 8, name='d_h3_conv')))

        h4 = ops.lrelu(self.d_bn4(ops.conv2d(h3, self.options['df_dim'] * 16, name='d_h4_conv')))

        h5 = ops.lrelu(self.d_bn5(ops.conv2d(h4, self.options['df_dim'] * 32, name='d_h5_conv')))

        # ADD TEXT EMBEDDING TO THE NETWORK
        reduced_text_embeddings = ops.lrelu(ops.linear(t_text_embedding, self.options['t_dim'], 'd_embedding'))
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 1)
        reduced_text_embeddings = tf.expand_dims(reduced_text_embeddings, 2)

        tiled_embeddings = tf.tile(reduced_text_embeddings, [1, 4, 4, 1], name='tiled_embeddings')

        h5_concat = tf.concat([h5, tiled_embeddings], 3, name='h5_concat')
        h5_new = ops.lrelu(
            self.d_bn6(ops.conv2d(h5_concat, self.options['df_dim'] * 8, 1, 1, 1, 1, name='d_h5_conv_new')))  # 4
        h6 = ops.linear(tf.reshape(h5_new, [self.options['batch_size'], -1]), 1, 'd_h6_lin')

        return tf.nn.sigmoid(h6), h6
