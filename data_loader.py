import os
from os.path import join
import argparse
import skipthoughts
import h5py


# use Fashion-166 dataset
def save_caption_vectors_cloth(data_dir):
    import time

    img_dir = join(data_dir, 'cloth/jpg')
    image_files = [f for f in os.listdir(img_dir) if 'jpg' in f]

    image_captions = {img_file: [] for img_file in image_files}

    caption_dir = join(data_dir, 'cloth/text_c166')
    class_dirs = []
    for i in range(1, 167):
        class_dir_name = 'class_%d' % (i)
        class_dirs.append(join(caption_dir, class_dir_name))

    for class_dir in class_dirs:
        caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
        for cap_file in caption_files:
            with open(join(class_dir, cap_file)) as f:
                captions = f.read().split('\n')
            img_file = cap_file.split(".")[0] + ".jpg"
            image_captions[img_file] += [cap for cap in captions if len(cap) > 0][0:5]
# load skip-thoughts pre-trained model

    model = skipthoughts.load_model()
    encoded_captions = {}

    for i, img in enumerate(image_captions):
        st = time.time()
        encoded_captions[img] = skipthoughts.encode(model, image_captions[img])
        print(i, len(image_captions), img)
        print("Seconds", time.time() - st)

    h = h5py.File(join(data_dir, 'cloth.hdf5'))
    for key in encoded_captions:
        h.create_dataset(key, data=encoded_captions[key])
    h.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='Data',
                        help='Data directory')
    parser.add_argument('--data_set', type=str, default='cloth',
                        help='Data Set : Flowers,')
    args = parser.parse_args()

    if args.data_set == 'cloth':
        save_caption_vectors_cloth(args.data_dir)
    else:
        print('dataset not found!')


if __name__ == '__main__':
    main()
