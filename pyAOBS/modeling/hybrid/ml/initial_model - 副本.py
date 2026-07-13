"""
初始模型生成模块
使用U-Net深度学习网络从走时数据生成初始速度模型
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import tqdm

class DoubleConv(nn.Module):
    """双重卷积块"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
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

class VelocityUNet(nn.Module):
    """用于速度模型生成的U-Net网络"""
    def __init__(self, n_channels: int = 3, n_classes: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class InitialModelGenerator:
    """初始速度模型生成器"""
    
    def __init__(self,
                 nx: int,
                 nz: int,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """初始化模型生成器
        
        Args:
            nx: x方向网格数
            nz: z方向网格数
            device: 计算设备
        """
        self.nx = nx
        self.nz = nz
        self.device = device
        self.model = VelocityUNet().to(device)
        
    def prepare_input(self,
                     travel_times: np.ndarray,
                     ray_paths: List[np.ndarray],
                     source_positions: np.ndarray,
                     receiver_positions: np.ndarray
                     ) -> torch.Tensor:
        """准备网络输入数据
        
        Args:
            travel_times: 走时数据
            ray_paths: 射线路径
            source_positions: 震源位置
            receiver_positions: 接收点位置
            
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
        return x.to(self.device)
        
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
            output = self.model(x)
            
        # 后处理
        velocity = output[0, 0].cpu().numpy()
        
        # 应用sigmoid激活函数将输出映射到(0,1)范围
        velocity = 1 / (1 + np.exp(-velocity))
        
        # 将值映射到合理的速度范围
        v_min, v_max = 1.5, 8.0  # 设置合理的速度范围
        velocity = v_min + (v_max - v_min) * velocity
        
        # 确保速度在合理范围内
        velocity = np.clip(velocity, v_min, v_max)
        
        return velocity
    
    def train(self,
             train_data: List[Tuple],
             valid_data: Optional[List[Tuple]] = None,
             n_epochs: int = 100,
             batch_size: int = 4,
             learning_rate: float = 1e-4,
             callback: Optional[callable] = None):
        """训练模型
        
        Args:
            train_data: 训练数据列表，每个元素为(travel_times, ray_paths, sources, receivers, velocity)
            valid_data: 验证数据列表(可选)
            n_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            callback: 回调函数，用于记录训练过程
        """
        # 创建数据加载器
        train_loader = DataLoader(
            VelocityDataset(train_data, self),
            batch_size=batch_size,
            shuffle=True
        )
        if valid_data:
            valid_loader = DataLoader(
                VelocityDataset(valid_data, self),
                batch_size=batch_size
            )
            
        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 训练循环
        best_valid_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
            for batch_idx, batch in enumerate(pbar):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                current_loss = loss.item()
                pbar.set_postfix({'train_loss': f'{current_loss:.6f}'})
                
                # 调用回调函数
                if callback:
                    callback(epoch, current_loss)
                
            train_loss /= len(train_loader)
            
            # 验证阶段
            if valid_data:
                self.model.eval()
                valid_loss = 0
                with torch.no_grad():
                    for batch in valid_loader:
                        inputs, targets = batch
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        valid_loss += loss.item()
                valid_loss /= len(valid_loader)
                
                print(f'Epoch {epoch+1}: train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}')
                
                # 早停
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print('Early stopping!')
                        # 加载最佳模型
                        self.model.load_state_dict(torch.load('best_model.pth'))
                        break
            else:
                print(f'Epoch {epoch+1}: train_loss={train_loss:.6f}')
                
class VelocityDataset(Dataset):
    """速度模型数据集"""
    
    def __init__(self, data: List[Tuple], generator: InitialModelGenerator):
        self.data = data
        self.generator = generator
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        travel_times, ray_paths, sources, receivers, velocity = self.data[idx]
        
        # 准备输入数据
        inputs = self.generator.prepare_input(
            travel_times, ray_paths,
            sources, receivers
        )
        
        # 移除多余的batch维度
        inputs = inputs.squeeze(0)
        
        # 准备目标数据
        targets = torch.from_numpy(velocity).float().unsqueeze(0)
        targets = targets.to(self.generator.device)
        
        return inputs, targets 