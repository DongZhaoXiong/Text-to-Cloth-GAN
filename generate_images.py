import tensorflow as tf
import numpy as np
import model
import argparse
from os.path import join
import h5py
import os
import imageio
import time


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--z_dim', type=int, default=100,
                        help='Noise Dimension')

    parser.add_argument('--t_dim', type=int, default=256,
                        help='Text feature dimension')

    parser.add_argument('--image_size', type=int, default=256,
                        help='Image Size')

    parser.add_argument('--gf_dim', type=int, default=64,
                        help='Number of conv in the first layer gen.')

    parser.add_argument('--df_dim', type=int, default=64,
                        help='Number of conv in the first layer discr.')

    parser.add_argument('--gfc_dim', type=int, default=1024,
                        help='Dimension of gen untis for for fully connected layer 1024')

    parser.add_argument('--caption_vector_length', type=int, default=2400,
                        help='Caption Vector Length')

    parser.add_argument('--data_dir', type=str, default="./Data",
                        help='Data Directory')

    parser.add_argument('--model_path', type=str, default='./Data/Models/',
                        help='Trained Model Path')

    parser.add_argument('--n_images', type=int, default=8,
                        help='Number of Images per Caption')

    parser.add_argument('--caption_thought_vectors', type=str, default='Data/sample_caption_vectors_single.hdf5',
                        help='Caption Thought Vector File')

    args = parser.parse_args()
    model_options = {
        'z_dim': args.z_dim,
        't_dim': args.t_dim,
        'batch_size': args.n_images,
        'image_size': args.image_size,
        'gf_dim': args.gf_dim,
        'df_dim': args.df_dim,
        'gfc_dim': args.gfc_dim,
        'caption_vector_length': args.caption_vector_length
    }

    tc_gan = model.TC_GAN(model_options)
    _, _, _, _, _ = tc_gan.build_model()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, args.model_path)

    input_tensors, outputs = tc_gan.build_generator()

    h = h5py.File(args.caption_thought_vectors)
    caption_vectors = np.array(h['vectors'])
    caption_image_dic = {}
    print("=======================================")
    print("start generate")
    begin = time.time()
    for cn, caption_vector in enumerate(caption_vectors):
        caption_images = []
        z_noise = np.random.uniform(-1, 1, [args.n_images, args.z_dim])
        caption = [caption_vector[0:args.caption_vector_length]] * args.n_images

        [gen_image] = sess.run([outputs['generator']],
                               feed_dict={
                                   input_tensors['t_real_caption']: caption,
                                   input_tensors['t_z']: z_noise,
                               })

        caption_images = [gen_image[i, :, :, :] for i in range(0, args.n_images)]
        caption_image_dic[cn] = caption_images
        print("Generated", cn)

    end = time.time()
    print("spend times:{}".format(end - begin))
    print("========================================")
    for f in os.listdir(join(args.data_dir, 'val_samples')):
        if os.path.isfile(f):
            os.unlink(join(args.data_dir, 'val_samples/' + f))

    for cn in range(0, len(caption_vectors)):
        caption_images = []
        for i, im in enumerate(caption_image_dic[cn]):
            caption_images.append(im)
            caption_images.append(np.zeros((256, 5, 3)))
        combined_image = np.concatenate(caption_images[0:-1], axis=1)
        imageio.imsave(join(args.data_dir, 'val_samples/{}_combined_image_{}.jpg'.format(args.caption_vector_length.split('.')[0][-6:],cn)), combined_image)


if __name__ == '__main__':
    main()
