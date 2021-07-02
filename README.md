# Text To Cloth Generative Adversarial Network(TC-GAN)

In [previous work](https://github.com/DongZhaoXiong/C-DCGAN.git), the generation of clothing images depended on the classification of datasets. In this work, Skip Thought Vectors is used to process the text description and TC-GAN algorithm is used to generate the cloth image. The experiment was completed using the TensorFlow platform. Here is the model structure.

<div align=center>
<img src="https://github.com/DongZhaoXiong/Text-to-Cloth-GAN/blob/main/TC-GAN%20network.png" > 
</div>


### Requirements

- Python 3.6.8
- Tensorflow 1.13.1
- h5py
- Theano 1.0.5
- scikit-learn 
- NLTK 3.6.2

### Datasets

- In order to better model training, this paper designed the Fathion -166 dataset.

- Fashion-166 is a text clothing image dataset based on the 

  [DeepFashion]: https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html

   dataset.

- Fashion-166 describes the text through subjective vision. At the same time, in order to reduce the influence of subjective consciousness on the objective results of the model, the method of multi-feature mixed labeling was adopted. Fashion-166 contains 9993 clothing images from 166 different categories. Clothing categories include: jacket, shorts, pants, plaid shirt, T-shirt, hoodie, leggings, sweatpants, suit, dress, turtleneck sweater, vest and other different categories. In addition to the ontology category, the text description of clothing features also includes different features such as material, appearance, neckline type, pattern type, adapted season, trouser waist type, trouser mouth type, sleeve length and trouser leg length. Some specific category statistics are as follows:

<div align=center>
<img src="https://github.com/DongZhaoXiong/Text-to-Cloth-GAN/blob/main/Fashion-166(part).png" > 
</div>

- You can download Fashion-166 dataset from[Fashion-166](https://drive.google.com/file/d/1RYBb-x6tgm7Jyo2ZuMS_6_QWkYnc3qD3/view?usp=sharing)
- The Fashon-166 dataset is stored in Data/cloth
- Download the pre-trained Skip Thoughts model and save it in Data/ SkipThoughts
- In the Data directory create a folder with the following structure:

~~~
└─Data
    ├─cloth
    ├─Models
    ├─samples
    ├─SkipThoughts
    └─val_samples
~~~

### Usage

- **Data processing**

~~~
python3 data_loader.py --data_set="cloth"
~~~

- **Train**

~~~
python3 train.py --data_set="cloth"
~~~

- **Generate cloth images using customize text descriptions**

- [ ] Write down the custom text description and save it in Data/sample_captions.txt to generate the Skip Thought Vector:

~~~
python3 generate_thought_vectors.py --caption_file="Data/sample_captions.txt"
~~~

- [ ] Use the Skip Thought Vectors obtained above to generate the image:

~~~
python3 generate_images.py --model_path=<trained model path> --n_image=<generate image number>
~~~

### Thanks

Thank you very much for the study of Reed et al([Generative Adversarial Text-to-Image Synthesis](http://arxiv.org/abs/1605.05396)), which provided good ideas

Thanks a lot for the code reference(https://github.com/paarthneekhara/text-to-image)

Thanks to the DeepFashion([DeepFashion](https://openaccess.thecvf.com/content_cvpr_2016/html/Liu_DeepFashion_Powering_Robust_CVPR_2016_paper.html)) dataset for providing clothing images to advance our labelling work

### Author
Dong Zhaoxiong
