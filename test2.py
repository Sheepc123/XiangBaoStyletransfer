# -*- coding:utf-8 -*-

import torch
from PIL import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from torchvision.utils import save_image
import os

# 设置图片大小
img_size = 512

# 设置训练设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Normalize(mean=[0.500, 0.500, 0.500],
                                 std=[0.229, 0.224, 0.225])


# 加载图片
def load_img(img_path):
    img = Image.open(img_path).convert('RGB')  # 使打开的图片通道为RGB格式
    img = img.resize((img_size, img_size))  # 对图片进行裁剪，为512x512
    img = transforms.ToTensor()(img)
    img = transform(img).unsqueeze(0)  # unsqueeze升维
    return img


# 显示图片
def show_img(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    return image


# 构建神经网络
class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features  # .features用于提取卷积层

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features


# 加载图片并移动到设备
content_img = load_img("D:/project/XBstyle-transfer/Style-transfer/datebase/content.jpeg").to(device)
style_img = load_img("D:/project/XBstyle-transfer/Style-transfer/datebase/style2.jpg").to(device)

# 初始化目标图像并将其移动到设备
target = content_img.clone().requires_grad_(True).to(device)  # 确保 target 在设备上

# 选择优化器
optimizer = torch.optim.Adam([target], lr=0.003)

# 加载VGG模型并将其移动到设备
vgg = VGGNet().to(device).eval()

# 设置训练次数和损失权重
total_step = 50000
style_weight = 100

# 设置tensorboard，用于可视化
writer = SummaryWriter("l")

# 提取内容和风格特征（将它们移动到设备上）
content_features = [x.detach() for x in vgg(content_img)]
style_features = [x.detach() for x in vgg(style_img)]

# 创建输出目录
output_dir = "output_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 开始训练
for step in range(total_step):
    # 获取目标图像的特征
    target_features = vgg(target)

    style_loss = 0
    content_loss = 0
    for f1, f2, f3 in zip(target_features, content_features, style_features):
        content_loss = torch.mean((f1 - f2) ** 2) + content_loss
        _, c, h, w = f1.size()  # 结果为torch.Size([1, 64, 512, 512])
        f1 = f1.view(c, h * w)  # 处理数据格式为后面gram计算
        f3 = f3.view(c, h * w)

        # 计算gram矩阵
        f1 = torch.mm(f1, f1.t())  # torch.mm()两个矩阵相乘
        f3 = torch.mm(f3, f3.t())
        style_loss = torch.mean((f1 - f3) ** 2) / (c * h * w) + style_loss

    # 总损失
    loss = content_loss + style_weight * style_loss

    # 更新目标图像
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar("loss", loss, step)

    # 每100步保存一次图像
    if step % 1000 == 0:
        denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
        img = target.clone().squeeze()
        img = denorm(img).clamp_(0, 1)
        img = show_img(img)
        writer.add_image("target", img, global_step=step)
        
        # 保存图像到磁盘
        save_image(img, os.path.join("D:/project/XBstyle-transfer/Style-transfer/style2output", f'step_{step}.png'))
        
    print("Step [{}/{}], Content Loss: {:.4f}, Style Loss: {:.4f}"
          .format(step, total_step, content_loss.item(), style_loss.item()))

# 关闭writer
writer.close()
