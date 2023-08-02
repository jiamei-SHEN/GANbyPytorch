# GANbyPytorch
An implementation of GAN using Pytorch.

![sample_from_testgan.png](samples%2Fsample_from_testgan.png)

## 中文说明
[test_gan.py](test_gan.py) 大体是根据 [B站课程](https://www.bilibili.com/video/BV1VT4y1e796/?spm_id_from=333.337.search-card.all.click&vd_source=c960549e8f49b2d8be21a60e6bbd2280)
自己写的，有一些小小的改动。

[gan_from_github.py](gan_from_github.py) 是 [github的一种实现](https://github.com/lyeoni/pytorch-mnist-GAN)，更新了一些老旧的用法。

[output_images_bs_32_latentdim_64](output_images_bs_32_latentdim_64) 和 [output_images_bs_100_latentdim_100](output_images_bs_100_latentdim_100)
是 test_gan.py 中模型不使用 dropout 以及使用 ReLu（而不是LeakyReLU），在 batchsize 为 100（32）， latent_dim 为 100（64）下的结果，效果不怎么样，但我不想删掉。

[samples](samples) 下的 sample_from_testgan.png  和 sample_from_githubgan.png 
分别是 [test_gan.py](test_gan.py) 和 [gan_from_github.py](gan_from_github.py) 的输出结果。


## English README
[test_gan.py](test_gan.py) is written following [Bilibili Video](https://www.bilibili.com/video/BV1VT4y1e796/?spm_id_from=333.337.search-card.all.click&vd_source=c960549e8f49b2d8be21a60e6bbd2280)
with a little modification.

[gan_from_github.py](gan_from_github.py) is copied from [an implementation from github](https://github.com/lyeoni/pytorch-mnist-GAN), and I update some deprecated places.

[output_images_bs_32_latentdim_64](output_images_bs_32_latentdim_64) and [output_images_bs_100_latentdim_100](output_images_bs_100_latentdim_100)
are results when the models in test_gan.py do not use dropout and ReLu（instead of LeakyReLU）, with batchsize 100（32）, latent_dim 100（64）.
These are bad results, but I don't want to delete them.

sample_from_testgan.png and sample_from_githubgan.png under [samples](samples)
are the output results of [test_gan.py](test_gan.py) and [gan_from_github.py](gan_from_github.py) respectively.