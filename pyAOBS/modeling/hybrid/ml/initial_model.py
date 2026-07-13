"""
初始模型生成模块
使用ResNet-U-Net深度学习网络从走时数据生成初始速度模型
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, TYPE_CHECKING
import tqdm
from torchvision.transforms.functional import rotate
from pytorch_msssim import SSIM
import os

class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出通道数不同，添加1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class DoubleConv(nn.Module):
    """双重卷积块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            ResBlock(in_channels, out_channels),
            ResBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class LocalFeatureExtractor(nn.Module):
    """局部特征提取模块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 添加空洞卷积以获取不同尺度的局部特征
        self.dilated_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.dilated_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 基础特征提取
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 多尺度特征提取
        d1 = self.dilated_conv1(out)
        d2 = self.dilated_conv2(out)
        
        # 特征融合
        out = out + d1 + d2
        
        # 应用注意力机制
        att = self.attention(out)
        out = out * att
        
        return out

class VelocityUNet(nn.Module):
    """用于速度模型生成的改进ResNet-U-Net网络"""
    def __init__(self, n_channels: int = 3, n_classes: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # 编码器部分
        self.inc = LocalFeatureExtractor(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        
        # 多尺度特征融合
        self.msf = nn.ModuleList([
            LocalFeatureExtractor(64, 64),
            LocalFeatureExtractor(128, 128),
            LocalFeatureExtractor(256, 256),
            LocalFeatureExtractor(512, 512)
        ])
        
        # 解码器部分
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        # 边缘检测分支
        self.edge_branch = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        
        # 最终输出层
        self.outc = nn.Sequential(
            nn.Conv2d(65, 32, 3, padding=1),  # 65 = 64 + 1 (特征 + 边缘)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 1)
        )

    def forward(self, x):
        # 编码路径
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 多尺度特征提取
        f1 = self.msf[0](x1)
        f2 = self.msf[1](x2)
        f3 = self.msf[2](x3)
        f4 = self.msf[3](x4)
        
        # 解码路径
        x = self.up1(x5, f4)
        x = self.up2(x, f3)
        x = self.up3(x, f2)
        x = self.up4(x, f1)
        
        # 边缘检测分支
        edge = self.edge_branch(x)
        
        # 特征融合
        x = torch.cat([x, edge], dim=1)
        logits = self.outc(x)
        
        return torch.sigmoid(logits), torch.sigmoid(edge)

class CustomLoss(nn.Module):
    """改进的多尺度损失函数"""
    def __init__(self, delta: float = 1.0, ssim_weight: float = 0.3, edge_weight: float = 0.2):
        super().__init__()
        self.delta = delta
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=1)
        
    def edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算边缘损失"""
        # 确保输入在[0,1]范围内
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        
        # 使用Sobel算子计算梯度
        sobel_x = torch.tensor([[-1, 0, 1], 
                               [-2, 0, 2], 
                               [-1, 0, 1]], device=pred.device).float()
        sobel_y = torch.tensor([[-1, -2, -1], 
                               [0, 0, 0], 
                               [1, 2, 1]], device=pred.device).float()
        
        # 添加通道维度和批次维度 [1, 1, 3, 3]
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        # 确保输入张量有正确的维度 [batch, channel, height, width]
        if len(pred.shape) == 3:
            pred = pred.unsqueeze(1)
        if len(target.shape) == 3:
            target = target.unsqueeze(1)
        
        # 计算预测和目标的梯度
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        # 计算梯度幅度
        pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2)
        target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2)
        
        # 归一化梯度
        pred_grad = pred_grad / (torch.max(pred_grad) + 1e-6)
        target_grad = target_grad / (torch.max(target_grad) + 1e-6)
        
        return F.l1_loss(pred_grad, target_grad)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, edge_pred: torch.Tensor = None) -> torch.Tensor:
        # 确保所有输入在[0,1]范围内
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)
        if edge_pred is not None:
            edge_pred = torch.clamp(edge_pred, 0, 1)
        
        # Huber Loss
        diff = pred - target
        abs_diff = torch.abs(diff)
        quadratic = torch.min(abs_diff, torch.tensor(self.delta).to(pred.device))
        huber_loss = 0.5 * quadratic.pow(2)
        huber_loss += self.delta * abs_diff - 0.5 * self.delta * self.delta
        huber_loss = huber_loss.mean()
        
        # SSIM Loss
        ssim_loss = 1 - self.ssim_module(pred, target)
        
        # 边缘损失
        edge_loss = self.edge_loss(pred, target)
        if edge_pred is not None:
            edge_loss = edge_loss + F.binary_cross_entropy(edge_pred, target)
        
        # 多尺度损失
        ms_loss = 0
        for scale in [1, 2, 4]:
            if scale > 1:
                pred_down = F.avg_pool2d(pred, scale)
                target_down = F.avg_pool2d(target, scale)
                # 确保下采样后的值也在[0,1]范围内
                pred_down = torch.clamp(pred_down, 0, 1)
                target_down = torch.clamp(target_down, 0, 1)
                ms_loss += self.edge_loss(pred_down, target_down)
        
        # 组合损失
        total_loss = (1 - self.ssim_weight - self.edge_weight) * huber_loss + \
                    self.ssim_weight * ssim_loss + \
                    self.edge_weight * (edge_loss + 0.1 * ms_loss)
        
        return total_loss

class InitialModelGenerator:
    """初始速度模型生成器"""
    
    def __init__(self,
                 nx: int,
                 nz: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 model_path: str = None):
        """初始化模型生成器
        
        Args:
            nx: x方向网格数
            nz: z方向网格数
            device: 计算设备
            model_path: 预训练模型的路径（可选）
        """
        self.nx = nx
        self.nz = nz
        self.device = device
        self.model = VelocityUNet().to(device)
        
        # 如果提供了模型路径，加载预训练模型
        if model_path is not None:
            self.load_model(model_path)
        
    def save_model(self, save_path: str, is_best: bool = False):
        """保存模型
        
        Args:
            save_path: 保存路径
            is_best: 是否为最佳模型
        """
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 准备保存的数据
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'nx': self.nx,
            'nz': self.nz,
            'is_best': is_best
        }
        
        # 保存模型
        torch.save(save_data, save_path)
        print(f"模型已保存到: {save_path}")
    
    def load_model(self, model_path: str):
        """加载预训练模型
        
        Args:
            model_path: 模型文件路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
            
        # 加载模型数据
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 验证模型参数
        if checkpoint['nx'] != self.nx or checkpoint['nz'] != self.nz:
            print(f"警告：加载的模型尺寸({checkpoint['nx']}, {checkpoint['nz']})与当前设置({self.nx}, {self.nz})不匹配")
        
        # 加载模型参数
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型: {model_path}")
        
        # 如果是最佳模型，打印提示
        if checkpoint.get('is_best', False):
            print("这是最佳性能模型")
            
    def prepare_input(self,
                     travel_times: np.ndarray,
                     ray_paths: List[np.ndarray],
                     source_positions: np.ndarray,
                     receiver_positions: np.ndarray,
                     to_device: bool = True
                     ) -> torch.Tensor:
        """准备网络输入数据
        
        Args:
            travel_times: 走时数据
            ray_paths: 射线路径
            source_positions: 震源位置
            receiver_positions: 接收点位置
            to_device: 是否将张量移动到指定设备
            
        Returns:
            input_tensor: 网络输入张量，形状为[channels, height, width]
        """
        # 初始化三个通道
        traveltime_field = np.zeros((self.nx, self.nz))
        ray_density = np.zeros((self.nx, self.nz))
        geometry = np.zeros((self.nx, self.nz))
        
        # 1. 构建走时场
        n_receivers_per_source = len(travel_times) // len(source_positions)
        for i, src in enumerate(source_positions):
            start_idx = i * n_receivers_per_source
            end_idx = (i + 1) * n_receivers_per_source
            for j in range(start_idx, end_idx):
                if j < len(travel_times):
                    tt = travel_times[j]
                    rcv = receiver_positions[j]
                    # 使用简单的线性插值
                    x = np.linspace(src[0], rcv[0], 100)
                    z = np.linspace(src[1], rcv[1], 100)
                    for k in range(len(x)):
                        ix = int(x[k] / (self.nx-1) * self.nx)
                        iz = int(z[k] / (self.nz-1) * self.nz)
                        if 0 <= ix < self.nx and 0 <= iz < self.nz:
                            traveltime_field[ix, iz] = tt
                    
        # 2. 构建射线密度场
        for ray in ray_paths:
            if ray is None or len(ray) < 2:
                continue
            for j in range(len(ray)-1):
                x1, z1 = ray[j]
                x2, z2 = ray[j+1]
                # 将坐标转换为网格索引
                ix1 = int(x1 / (self.nx-1) * self.nx)
                iz1 = int(z1 / (self.nz-1) * self.nz)
                ix2 = int(x2 / (self.nx-1) * self.nx)
                iz2 = int(z2 / (self.nz-1) * self.nz)
                # 使用Bresenham算法标记射线路径
                dx = abs(ix2 - ix1)
                dz = abs(iz2 - iz1)
                if dx > dz:
                    steps = dx
                else:
                    steps = dz
                if steps > 0:
                    x_inc = (ix2 - ix1) / steps
                    z_inc = (iz2 - iz1) / steps
                    x = ix1
                    z = iz1
                    for _ in range(steps):
                        if 0 <= int(x) < self.nx and 0 <= int(z) < self.nz:
                            ray_density[int(x), int(z)] += 1
                        x += x_inc
                        z += z_inc
                        
        # 3. 构建震源接收点位置场
        for src in source_positions:
            ix = int(src[0] / (self.nx-1) * self.nx)
            iz = int(src[1] / (self.nz-1) * self.nz)
            if 0 <= ix < self.nx and 0 <= iz < self.nz:
                geometry[ix, iz] = 1
        for rcv in receiver_positions:
            ix = int(rcv[0] / (self.nx-1) * self.nx)
            iz = int(rcv[1] / (self.nz-1) * self.nz)
            if 0 <= ix < self.nx and 0 <= iz < self.nz:
                geometry[ix, iz] = 2
                
        # 4. 归一化
        if np.max(traveltime_field) > np.min(traveltime_field):
            traveltime_field = (traveltime_field - np.min(traveltime_field)) / (np.max(traveltime_field) - np.min(traveltime_field))
        if np.max(ray_density) > 0:
            ray_density = ray_density / np.max(ray_density)
        
        # 5. 组合为张量 [channels, height, width]
        x = np.stack([traveltime_field, ray_density, geometry])
        x = torch.from_numpy(x).float()
        if to_device:
            x = x.to(self.device)
        return x
        
    def generate(self,
                travel_times: np.ndarray,
                ray_paths: List[np.ndarray],
                source_positions: np.ndarray,
                receiver_positions: np.ndarray
                ) -> np.ndarray:
        """生成初始速度模型
        
        Args:
            travel_times: 走时数据
            ray_paths: 射线路径
            source_positions: 震源位置
            receiver_positions: 接收点位置
            
        Returns:
            velocity: 速度模型
        """
        # 准备输入数据
        x = self.prepare_input(travel_times, ray_paths,
                             source_positions, receiver_positions)
        
        # 添加batch维度
        x = x.unsqueeze(0)
        
        # 前向计算
        self.model.eval()
        with torch.no_grad():
            pred_velocity, _ = self.model(x)  # 只使用速度预测，忽略边缘预测
        
        # 后处理
        velocity = pred_velocity[0, 0].cpu().numpy()
        
        # 反归一化到实际速度范围
        v_min, v_max = 1.5, 8.0
        velocity = velocity * (v_max - v_min) + v_min
        
        return velocity
    
    def train(self,
             train_data: List[Tuple],
             valid_data: Optional[List[Tuple]] = None,
             n_epochs: int = 200,
             batch_size: int = 16,
             learning_rate: float = 2e-4,
             callback: Optional[callable] = None,
             save_dir: str = 'saved_models'):
        """训练模型
        
        Args:
            train_data: 训练数据列表
            valid_data: 验证数据列表(可选)
            n_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            callback: 回调函数
            save_dir: 模型保存目录
        """
        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 设置CUDA相关参数
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            num_workers = 0  # 在CUDA模式下使用0个worker以避免多进程问题
        else:
            num_workers = 4
            
        # 创建数据加载器
        train_loader = DataLoader(
            VelocityDataset(train_data, self, augment=True),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        if valid_data:
            valid_loader = DataLoader(
                VelocityDataset(valid_data, self, augment=False),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True
            )
            
        # 定义优化器和损失函数
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        criterion = CustomLoss()
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 训练循环
        best_valid_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                # 将数据移动到正确的设备
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                pred_velocity, pred_edge = self.model(inputs)  # 获取速度场和边缘预测
                loss = criterion(pred_velocity, targets, pred_edge)  # 传入边缘预测
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                current_loss = loss.item()
                pbar.set_postfix({'train_loss': f'{current_loss:.6f}'})
                
                if callback:
                    callback(epoch, current_loss)
            
            train_loss /= len(train_loader)
            
            # 验证阶段
            if valid_data:
                self.model.eval()
                valid_loss = 0
                with torch.no_grad():
                    for inputs, targets in valid_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        pred_velocity, pred_edge = self.model(inputs)
                        loss = criterion(pred_velocity, targets, pred_edge)
                        valid_loss += loss.item()
                
                valid_loss /= len(valid_loader)
                
                # 更新学习率
                scheduler.step(valid_loss)
                
                print(f'Epoch {epoch+1}: train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}')
                
                # 早停
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    patience_counter = 0
                    self.save_model(
                        os.path.join(save_dir, 'best_model.pth'),
                        is_best=True
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print('Early stopping!')
                        # 加载最佳模型
                        checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        break
            else:
                print(f'Epoch {epoch+1}: train_loss={train_loss:.6f}')
                if epoch % 10 == 0:
                    self.save_model(
                        os.path.join(save_dir, f'model_epoch_{epoch+1}.pth')
                    )
        
        # 保存最终模型
        self.save_model(
            os.path.join(save_dir, 'final_model.pth')
        )

class VelocityDataset(Dataset):
    """速度模型数据集"""
    
    def __init__(self, data: List[Tuple], generator: 'InitialModelGenerator', augment: bool = True):
        self.data = data
        self.generator = generator
        self.augment = augment
        
    def __len__(self):
        return len(self.data)
    
    def augment_data(self, inputs: torch.Tensor, targets: torch.Tensor):
        """数据增强"""
        # 随机旋转
        if np.random.random() < 0.5:
            angle = int(np.random.choice([-10, -5, 5, 10]).item())  # 确保返回整数
            inputs = rotate(inputs, angle)
            targets = rotate(targets, angle)
        
        # 随机水平翻转
        if np.random.random() < 0.5:
            inputs = torch.flip(inputs, [2])
            targets = torch.flip(targets, [2])
            
        # 添加随机噪声
        if np.random.random() < 0.3:
            noise = torch.randn_like(inputs) * 0.02
            inputs = inputs + noise
            inputs = torch.clamp(inputs, 0, 1)
            
        return inputs, targets
        
    def __getitem__(self, idx):
        travel_times, ray_paths, sources, receivers, velocity = self.data[idx]
        
        # 准备输入数据，保持在CPU上
        inputs = self.generator.prepare_input(
            travel_times, ray_paths,
            sources, receivers,
            to_device=False  # 保持在CPU上
        )
        
        # 准备目标数据
        targets = torch.from_numpy(velocity).float()
        
        # 确保输入和目标具有相同的空间维度
        if inputs.shape[-2:] != targets.shape:
            targets = F.interpolate(
                targets.unsqueeze(0).unsqueeze(0),
                size=inputs.shape[-2:],
                mode='bilinear',
                align_corners=True
            ).squeeze(0).squeeze(0)
        
        # 归一化目标数据到0-1范围
        targets = (targets - 1.5) / (8.0 - 1.5)  # 使用与generate方法相同的速度范围
        # 确保目标值在[0,1]范围内
        targets = torch.clamp(targets, 0, 1)
        targets = targets.unsqueeze(0)  # 添加通道维度
        
        # 数据增强
        if self.augment:
            inputs, targets = self.augment_data(inputs, targets)
            # 确保数据增强后的值仍在[0,1]范围内
            inputs = torch.clamp(inputs, 0, 1)
            targets = torch.clamp(targets, 0, 1)
        
        return inputs, targets
                    