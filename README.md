# remove-watermark

A model based on Pix2Pix (https://arxiv.org/abs/1611.07004) to remove watermark from images.
The implementation of Pix2Pix is based on https://github.com/eriklindernoren/PyTorch-GAN.

## implementation

The model takes a 256² crop image as input and outputs an 256² image with the watermark removed.
To transform a higher resolution image the model takes 256² crops of the images and stitches them together, this can
lead to weird artificats around the borders of the squares.

An additional perception loss is added to the generator based on vgg11. This can probably be changed to vgg19 but due
to computational limits this is not done.

## dataset

The dataset I used for this is not publicly available and thus I won't share it (also due to legal reasons).
The train dataset consists of around 20k Shutterstock images and their watermarked counterpart.
The validation set consists of around 200 images. Some sample images are shown below

## training

The model was trained for about 100 hours on a gtx 970.

## results

After 100 epochs these are the results. This can probably be improved if we train the model longer. The original images are not shown because they are copyrighted. More results can be found in the results folder.

![res 2 of validation data](results/epoch100/res1.jpg?raw=true "example 2")
![res 3 of validation data](results/epoch100/res2.jpg?raw=true "example 3")
![res 4 of validation data](results/epoch100/res3.jpg?raw=true "example 4")
![res 5 of validation data](results/epoch100/res4.jpg?raw=true "example 5")
