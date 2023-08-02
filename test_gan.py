"""基于MNIST实现生成对抗网络GAN"""
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image

image_size = [1, 28, 28]


# 生成器类
class Generator(nn.Module):

    # 定义一系列模块
    # in_dim:z的输入维度
    def __init__(self, input_dim):
        # 继承自nn.Module，所以需要对父类实例化
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),  # 全连接，64随便写的
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 128),  # 全连接，慢慢扩大
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, torch.prod(torch.Tensor(image_size), dtype=torch.int32)),  # 把特征映射到图像大小
            nn.Tanh(),
        )

    # 把模块串联起来，生成照片
    def forward(self, z):
        # z的形状：[batchsize, latent_dim]
        # 1:通道数 28 28：高度和宽度
        # 把z直接传入到model之中就好了
        output = self.model(z)
        image = output.reshape(z.shape[0], *image_size)  # image_size是一个列表，加*变成元组的形式

        return image


# 判别器类
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        input_size = torch.prod(torch.tensor(image_size)).item()  # 计算输入大小
        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 要弄出一个概率，所以需要用Sigmoid
        )

    # 判别器输入图片，输出概率
    def forward(self, input_image):
        # image的格式：[batchsize, 1, 28, 28]
        prob = self.model(input_image.reshape(input_image.size(0), -1))  # 把4维的image变成2维的
        return prob  # 返回概率


# Training
# 定义输出图片的文件夹路径
output_folder = "output_images"
os.makedirs(output_folder, exist_ok=True)
# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = torchvision.datasets.MNIST("mnist_data", train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                         torchvision.transforms.Resize(28),
                                         torchvision.transforms.ToTensor(),
                                         torchvision.transforms.Normalize(mean=[0.5], std=[0.5]),
                                     ]))

# dataLoader把dataset里每个样本构成一个minibatch，后面去做批训练
batch_size = 100
dataLoader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# 要2个优化器，分别对生成器和判别器的参数进行优化
# 实例化Generator
latent_dim = 100
generator = Generator(latent_dim).to(device)
g_optimizer = torch.optim.Adam(params=generator.parameters(), lr=0.0001)
# 实例化Discriminator
discriminator = Discriminator().to(device)
d_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=0.0001)

# loss function
loss_fn = nn.BCELoss()

# 以上完成了定义部分，下面正式开始训练
num_epoch = 200
# 进行多少个epoch
for epoch in range(1, num_epoch+1):
    print(f'epoch {epoch}/{num_epoch}')
    # 对dataLoader进行枚举遍历
    for i, mini_batch in enumerate(dataLoader):
        gt_image, _ = mini_batch

        # 将输入数据移到相同设备上
        gt_image = gt_image.to(device)

        z = torch.randn(batch_size, latent_dim).to(device)
        pred_images = generator(z)

        # 优化生成器，所以target=1
        g_optimizer.zero_grad()
        target_1 = torch.ones(batch_size, 1).to(device)
        g_loss = loss_fn(discriminator(pred_images), target_1)
        g_loss.backward()
        g_optimizer.step()

        # 优化判别器
        d_optimizer.zero_grad()
        target_0 = torch.zeros(batch_size, 1).to(device)
        # 对生成器pre_image部分调用detach函数把这部分从计算图中分离出来，不需要记录生成器部分的梯度
        d_loss = 0.5 * (
                    loss_fn(discriminator(pred_images.detach()), target_0) + loss_fn(discriminator(gt_image), target_1))
        d_loss.backward()
        d_optimizer.step()

        '''if i % 1500 == 0:
            for index, image in enumerate(pred_images):
                torchvision.utils.save_image(image, f"image_{index}.png")'''

        '''if epoch % 10 == 0 and i % 1000 == 0:
            images_to_save = pred_images[:32]  # 只取前32张图片
            combined_image = torchvision.utils.make_grid(images_to_save, nrow=8, padding=2,
                                                         normalize=True)  # 将32张图片合并成一张大图
            torchvision.utils.save_image(combined_image,
                                         os.path.join(output_folder, f"combined_image_epoch_{epoch}.png"))
            print(f'写入 epoch {epoch} 结果')'''

# 在训练完成后，保存生成器和判别器的状态字典
torch.save(generator.state_dict(), f'generator_bs_{batch_size}_epoch_{num_epoch}.pth')
torch.save(discriminator.state_dict(), f'discriminator_bs_{batch_size}_epoch_{num_epoch}.pth')

with torch.no_grad():
    test_z = torch.randn(batch_size, latent_dim).to(device)
    generated = generator(test_z)
    if not os.path.exists('./samples'):
        os.makedirs('./samples')
    save_image(generated.view(generated.size(0), 1, 28, 28), './samples/sample_' + '.png')