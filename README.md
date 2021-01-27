# DCGAN in tensorflow1
###### DCGAN in tensorflow2 : coming soon...
###### Tensorflow1 implementation of DCGAN.
###### [paper](https://arxiv.org/pdf/1511.06434.pdf)
----------------
## Prerequisites
- Python 3.6
- Tensorflow 1.14.0
- Opencv-contrib-python 4.5.1.48

----------------
## Generator architecture of DCGAN

![](https://github.com/Hwa-Jong/DCGAN/blob/main/img/Generator(DCGAN).png)

----------------
## Usage

1. Preparing your dataset. (In my case, I prepared images in the "dataset" directory.)

2. Train the model.
> ```
> python3 main.py --dataset_dir=<your dataset path>
> ```
> ex)
> ```
> python3 main.py --dataset_dir=dataset --epochs=100(default) --batch_size=128(default)
> ```
3. Using pre-trained model
> ```
> python3 main.py --dataset_dir=<your dataset path> --load_path=<model path>
> ```
> ex)
> ```
> python3 main.py --dataset_dir=dataset --epochs=100(default) --batch_size=128(default) --load_path=results/0001_DCGAN_batch-128_epoch-100/ckpt/model.ckpt-50
> ```
4. Generate images
> ```
> python3 generate.py --load_path=<model path>
> ```
> ex)
> ```
> python3 generate.py --load_path=results/0001_DCGAN_batch-128_epoch-100/ckpt/model.ckpt-50 --generate_num=16(default) --seed=22222(default)
> ```

----------------
## Result 
* Using 30k images of CelebA dataset.
> ### 1 epoch
> ![](https://github.com/Hwa-Jong/DCGAN-in-tf1/blob/main/img/fake%2000001epoc.png)

> ### 10 epochs
> ![](https://github.com/Hwa-Jong/DCGAN-in-tf1/blob/main/img/fake%2000010epoc.png)

> ### 50 epochs
> ![](https://github.com/Hwa-Jong/DCGAN-in-tf1/blob/main/img/fake%2000050epoc.png)

> ### 80 epochs
> ![](https://github.com/Hwa-Jong/DCGAN-in-tf1/blob/main/img/fake%2000080epoc.png)

> ### 100 epochs
> ![](https://github.com/Hwa-Jong/DCGAN-in-tf1/blob/main/img/fake%2000100epoc.png)

