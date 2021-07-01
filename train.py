import tensorflow as tf

import numpy as np

import model

import argparse

from os.path import join

import h5py

import image_processing

import random

import os

import shutil

import imageio


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--z_dim', type=int, default=100,

                        help='Noise dimension')

    parser.add_argument('--t_dim', type=int, default=256,

                        help='Text feature dimension')

    parser.add_argument('--batch_size', type=int, default=64,

                        help='Batch Size')

    parser.add_argument('--image_size', type=int, default=256,

                        help='Image Size a, a x a')

    parser.add_argument('--gf_dim', type=int, default=64,

                        help='Number of conv in the first layer gen.')

    parser.add_argument('--df_dim', type=int, default=64,

                        help='Number of conv in the first layer discr.')

    parser.add_argument('--gfc_dim', type=int, default=1024,

                        help='Dimension of gen untis for for fully connected layer 1024')

    parser.add_argument('--caption_vector_length', type=int, default=2400,

                        help='Caption Vector Length')

    parser.add_argument('--data_dir', type=str, default="Data",

                        help='Data Directory')

    parser.add_argument('--learning_rate', type=float, default=0.0002,

                        help='Learning Rate')

    parser.add_argument('--beta1', type=float, default=0.5,

                        help='Momentum for Adam Update')

    parser.add_argument('--epochs', type=int, default=100,

                        help='Max number of epochs')

    parser.add_argument('--save_every', type=int, default=30,

                        help='Save Model/Samples every x iterations over batches')

    parser.add_argument('--resume_model', type=str, default=None,
                        help='Pre-Trained Model Path, to resume from')

    parser.add_argument('--data_set', type=str, default="cloth",
                        help='Dat set: cloth')

    args = parser.parse_args()

    model_options = {

        'z_dim': args.z_dim,

        't_dim': args.t_dim,

        'batch_size': args.batch_size,

        'image_size': args.image_size,

        'gf_dim': args.gf_dim,

        'df_dim': args.df_dim,

        'gfc_dim': args.gfc_dim,

        'caption_vector_length': args.caption_vector_length

    }
# set GPU options
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.allow_soft_placement = True
# save train log,use tensorboard to visual
    model_summaries_dir = "./logs"

    tc_gan = model.TC_GAN(model_options)

    input_tensors, variables, loss, outputs, checks = tc_gan.build_model()

    d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1).minimize(loss['d_loss'],
                                                                                    var_list=variables['d_vars'])

    g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1).minimize(loss['g_loss'],
                                                                                    var_list=variables['g_vars'])

    global_step_tensor = tf.Variable(1, trainable=False, name='global_step')
    merged = tf.summary.merge_all()
    sess = tf.InteractiveSession()

    summary_writer = tf.summary.FileWriter(model_summaries_dir, sess.graph)
    tf.global_variables_initializer().run()

    saver = tf.train.Saver(max_to_keep=10000)

    if args.resume_model:
        saver.restore(sess, args.resume_model)

    global_step = global_step_tensor.eval()
    gs_assign_op = global_step_tensor.assign(global_step)

    loaded_data = load_training_data(args.data_dir, args.data_set)

    for i in range(args.epochs):

        batch_no = 0

        while batch_no * args.batch_size < loaded_data['data_length']:

            real_images, wrong_images, caption_vectors, z_noise, image_files = get_training_batch(batch_no,
                                                                                                  args.batch_size,
                                                                                                  args.image_size,
                                                                                                  args.z_dim,
                                                                                                  args.caption_vector_length,
                                                                                                  args.data_dir,
                                                                                                  args.data_set,
                                                                                                  loaded_data)

            #  DISCRIMINATOR UPDATE

            check_ts = [checks['d_loss1'], checks['d_loss2'], checks['d_loss3']]
            feed = {
                input_tensors['t_real_image'].name: real_images,
                input_tensors['t_wrong_image'].name: wrong_images,
                input_tensors['t_real_caption'].name: caption_vectors,
                input_tensors['t_z'].name: z_noise,
            }

            _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,

                                                  feed_dict=feed)
            print("D loss-1 [loss for real images] : {} \n"
                  "D loss-2 [loss for wrong images] : {} \n"
                  "D loss-3 [loss for fake images] : {} \n"
                  "D total loss : {}".format(d1,d2,d3,d_loss))

            # GENERATOR UPDATE

            _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],

                                      feed_dict=feed)

            # GENERATOR UPDATE TWICE, to make sure d_loss does not go to 0

            _, summary, g_loss, gen = sess.run([g_optim, merged, loss['g_loss'], outputs['generator']],

                                               feed_dict=feed)

            summary_writer.add_summary(summary, global_step)

            print("\nLOSSES\nDiscriminator Loss: {}\nGenerator Loss: {}\nBatch Numer: {}\nEpoch: {}\nTotal Batches "
                  "per epoch: {}".format(d_loss, g_loss, batch_no, i, len(loaded_data['image_list']) /
                                         args.batch_size))

            global_step += 1
            sess.run(gs_assign_op)
            batch_no += 1

            if (batch_no % args.save_every) == 0:
                print("Saving Images, Model")

                save_for_vis(args.data_dir, real_images, gen, image_files)

                save_path = saver.save(sess, "Data_1/Models/latest_model_{}_temp.ckpt".format(args.data_set))

        if i % 5 == 0:
            save_path = saver.save(sess, "Data_1/Models/model_after_{}_epoch_{}.ckpt".format(args.data_set, i))


def load_training_data(data_dir, data_set):
    if data_set == 'cloth':

        h = h5py.File(join(data_dir, 'cloth.hdf5'))

        cloth_captions = {}

        for ds in h.items():
            cloth_captions[ds[0]] = np.array(ds[1])

        image_list = [key for key in cloth_captions]

        image_list.sort()

        training_image_list = image_list

        random.shuffle(training_image_list)

        return {

            'image_list': training_image_list,

            'captions': cloth_captions,

            'data_length': len(training_image_list)

        }


def save_for_vis(data_dir, real_images, generated_images, image_files):
    shutil.rmtree(join(data_dir, 'samples'))

    os.makedirs(join(data_dir, 'samples'))
    for i in range(0, real_images.shape[0]):
        real_images_255 = (real_images[i, :, :, :])
        imageio.imsave(join(data_dir, 'samples_1/{}_{}'.format(i, image_files[i].split("/")[-1])), real_images_255)
        fake_images_255 = (generated_images[i, :, :, :])
        imageio.imsave(join(data_dir, 'samples_1/fake_image_{}.jpg'.format(i)), fake_images_255)


def get_training_batch(batch_no, batch_size, image_size, z_dim,

                       caption_vector_length, data_dir, data_set, loaded_data=None):
    if data_set == 'cloth':

        real_images = np.zeros((batch_size, 256, 256, 3))

        wrong_images = np.zeros((batch_size, 256, 256, 3))

        captions = np.zeros((batch_size, caption_vector_length))

        cnt = 0

        image_files = []

        for i in range(batch_no * batch_size, batch_no * batch_size + batch_size):
            idx = i % len(loaded_data['image_list'])

            image_file = join(data_dir, 'cloth/jpg/' + loaded_data['image_list'][idx])

            image_array = image_processing.load_image_array(image_file, image_size)

            real_images[cnt, :, :, :] = image_array

            # Improve this selection of wrong image

            wrong_image_id = random.randint(0, len(loaded_data['image_list']) - 1)

            wrong_image_file = join(data_dir, 'cloth/jpg/' + loaded_data['image_list'][wrong_image_id])

            wrong_image_array = image_processing.load_image_array(wrong_image_file, image_size)

            wrong_images[cnt, :, :, :] = wrong_image_array

            random_caption = random.randint(0, 4)
            captions[cnt, :] = loaded_data['captions'][loaded_data['image_list'][idx]][random_caption][
                               0:caption_vector_length]

            image_files.append(image_file)

            cnt += 1

        z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])

        return real_images, wrong_images, captions, z_noise, image_files


if __name__ == '__main__':
    main()
