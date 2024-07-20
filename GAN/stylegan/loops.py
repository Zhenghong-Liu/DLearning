import torch
from torch.autograd import grad
from config import device, w_dimensions, z_dimensions


def generate_noise(batch_size, noise_channels, device):
    return torch.randn([batch_size, noise_channels], device=device)

def r1_penalty(real_batch_predictions, real_batch):
    gradients = grad(
        outputs=real_batch_predictions.sum(), inputs=real_batch, create_graph=True
    )[0]
    return gradients.pow(2).sum(dim=[1, 2, 3]).mean() #这里的dim=[1, 2, 3]是因为我们要对每个样本的每个像素点的梯度进行求和，然后再求平均

def gradient_penalty(discriminator, real_batch, fake_batch, batch_size, resolution, alpha):
    ratio = torch.rand([batch_size, 1, 1, 1], device=device) #生成[0, 1)的均匀分布
    interpolation = torch.lerp(real_batch, fake_batch, ratio) #torch.lerp(star, end, weight) : 返回结果是out= start+ (end-start) * weight

    interpolation_predictions = discriminator(interpolation, resolution, alpha)
    gradients = grad(
        interpolation_predictions, interpolation,
        torch.ones_like(interpolation_predictions),
        create_graph=True
    )[0]

    return ((gradients.norm(2, dim=[1, 2, 3]) - 1) ** 2).mean()

loss_function = torch.nn.BCEWithLogitsLoss() #二分类交叉熵损失函数

def train_discriminator(
        mapping_network,
        generator,
        discriminator,
        real_batch,
        optimizer,
        penalty_factor,
        batch_size,
        resolution,
        alpha):

    real_batch.requires_grad = True
    real_batch_scores = discriminator(real_batch, resolution, alpha)

    #Mixing regularization
    fake_batch_w1 = mapping_network(generate_noise(batch_size, z_dimensions, device))
    fake_batch_w2 = mapping_network(generate_noise(batch_size, z_dimensions, device))
    #生成交叉的位置
    mixing_indices = torch.randint(1, w_dimensions, [batch_size,])
    mixing_indices = torch.round(torch.rand([batch_size,])) * mixing_indices #50%的概率交叉，50%的概率不交叉
    #我们想要不交叉的话，保留的是w1
    mixing_indices = (z_dimensions - mixing_indices).to(torch.int)

    #交叉
    fake_batch_w = torch.empty_like(fake_batch_w1)
    for i, index in enumerate(mixing_indices):
        fake_batch_w[i, :index] = fake_batch_w1[i, :index]
        fake_batch_w[i, index:] = fake_batch_w2[i, index:]

    fake_batch = generator(fake_batch_w, resolution, alpha)
    fake_batch_scores = discriminator(fake_batch.detach(), resolution, alpha) #以我个人的理解，这里应该detach，因为我们不需要计算生成器的梯度（但是参考代码里面是没有的）

    penalty = r1_penalty(real_batch_scores, real_batch)
    real_batch.requires_grad_(False)
    discriminator_loss = (
        loss_function(real_batch_scores, torch.ones_like(real_batch_scores))
        + loss_function(fake_batch_scores, torch.zeros_like(fake_batch_scores))
        + penalty_factor * penalty
    )

    optimizer.zero_grad()
    discriminator_loss.backward()
    optimizer.step()

    # return real_batch_scores.mean(), fake_batch_scores.mean(), discriminator_loss.detach()
    return real_batch_scores.mean().item(), fake_batch_scores.mean().item(), discriminator_loss.item()

def train_generator(mapping_network, generator, discriminator, optimizer, batch_size, resolution, alpha):
    #Mixing regularization
    # Mixing regularization
    fake_batch_w1 = mapping_network(generate_noise(batch_size, z_dimensions, device))
    fake_batch_w2 = mapping_network(generate_noise(batch_size, z_dimensions, device))
    # 生成交叉的位置
    mixing_indices = torch.randint(1, w_dimensions, [batch_size, ])
    mixing_indices = torch.round(torch.rand([batch_size, ])) * mixing_indices  # 50%的概率交叉，50%的概率不交叉
    # 我们想要不交叉的话，保留的是w1
    mixing_indices = (z_dimensions - mixing_indices).to(torch.int)

    # 交叉
    fake_batch_w = torch.empty_like(fake_batch_w1)
    for i, index in enumerate(mixing_indices):
        fake_batch_w[i, :index] = fake_batch_w1[i, :index]
        fake_batch_w[i, index:] = fake_batch_w2[i, index:]

    fake_batch = generator(fake_batch_w, resolution, alpha)
    fake_batch_scores = discriminator(fake_batch, resolution, alpha)

    generator_loss = loss_function(fake_batch_scores, torch.ones_like(fake_batch_scores))

    optimizer.zero_grad()
    generator_loss.backward()
    optimizer.step()

    # return generator_loss.detach() #detach是生成新的向量，这个新的向量不再需要计算梯度。
    #这里用loss.item()的写法也是可以的, 比detach写法好一些，因为item()只是返回一个标量，而detach()是返回一个新的tensor
    return generator_loss.item()


if __name__ == '__main__':
    # =============================================
    # 判断detach方法是在原始数据上修改，还是生成了新的数据
    # =============================================
    a = torch.randn([1, 2])
    print(a)
    b = a.detach()
    print(b)

    #判断a和b是不是同一个tensor
    print(a is b)






