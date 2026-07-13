import unittest
import numpy as np
import torch
import matplotlib.pyplot as plt
from initial_model import InitialModelGenerator, VelocityUNet
import os

class TestInitialModel(unittest.TestCase):
    def setUp(self):
        # 设置测试参数
        self.nx = 100  # x方向网格数
        self.nz = 50   # z方向网格数
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 创建模型生成器实例
        self.generator = InitialModelGenerator(self.nx, self.nz, self.device)
        
        # 创建输出目录
        self.output_dir = 'test_outputs'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def visualize_tensor(self, tensor, title, filename):
        """可视化张量数据"""
        plt.figure(figsize=(15, 5))
        
        if len(tensor.shape) == 3:  # 多通道数据
            n_channels = tensor.shape[0]
            for i in range(n_channels):
                plt.subplot(1, n_channels, i+1)
                # 使用detach()分离梯度
                plt.imshow(tensor[i].detach().cpu().numpy().T, cmap='jet')
                plt.colorbar()
                if i == 0:
                    plt.title('走时场')
                elif i == 1:
                    plt.title('射线密度场')
                elif i == 2:
                    plt.title('几何场')
                else:
                    plt.title(f'Channel {i+1}')
        else:  # 单通道数据
            # 使用detach()分离梯度
            plt.imshow(tensor.detach().cpu().numpy().T, cmap='jet')
            plt.colorbar()
            plt.title(title)
            
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        
    def test_model_architecture(self):
        """测试U-Net模型架构"""
        model = VelocityUNet()
        # 测试输入维度
        test_input = torch.randn(1, 3, self.nx, self.nz)
        output = model(test_input)
        self.assertEqual(output.shape, (1, 1, self.nx, self.nz))
        
        # 可视化随机输入和输出
        self.visualize_tensor(test_input[0], 'Random Input Data', 'test_random_input.png')
        self.visualize_tensor(output[0], 'Random Output Data', 'test_random_output.png')
        
    def test_prepare_input(self):
        """测试输入数据准备"""
        # 创建模拟数据
        n_rays = 10
        travel_times = np.random.rand(n_rays)
        ray_paths = [np.array([[0, 0], [self.nx-1, self.nz-1]]) for _ in range(n_rays)]
        source_positions = np.array([[0, 0] for _ in range(n_rays)])
        receiver_positions = np.array([[self.nx-1, self.nz-1] for _ in range(n_rays)])
        
        # 测试输入准备
        input_tensor = self.generator.prepare_input(
            travel_times, ray_paths, source_positions, receiver_positions
        )
        
        # 验证输出维度 [channels, height, width]
        self.assertEqual(input_tensor.shape, (3, self.nx, self.nz))
        self.assertEqual(input_tensor.device.type, self.device)
        
        # 验证数据范围
        self.assertTrue(torch.all(input_tensor >= 0))  # 所有值应该非负
        self.assertTrue(torch.all(input_tensor[0] <= 1))  # 走时场已归一化
        self.assertTrue(torch.all(input_tensor[1] <= 1))  # 射线密度已归一化
        self.assertTrue(torch.all(input_tensor[2] <= 2))  # 几何场最大值为2
        
        # 可视化准备的输入数据
        self.visualize_tensor(input_tensor, 'Prepared Input Data', 'test_prepared_input.png')
        
    def test_model_generation(self):
        """测试模型生成功能"""
        # 创建模拟数据
        n_rays = 10
        travel_times = np.random.rand(n_rays)
        ray_paths = [np.array([[0, 0], [self.nx-1, self.nz-1]]) for _ in range(n_rays)]
        source_positions = np.array([[0, 0] for _ in range(n_rays)])
        receiver_positions = np.array([[self.nx-1, self.nz-1] for _ in range(n_rays)])
        
        # 测试速度模型生成
        velocity = self.generator.generate(
            travel_times, ray_paths, source_positions, receiver_positions
        )
        
        # 验证输出维度和类型
        self.assertEqual(velocity.shape, (self.nx, self.nz))
        self.assertTrue(isinstance(velocity, np.ndarray))
        
        # 可视化生成的速度模型
        plt.figure(figsize=(10, 6))
        plt.imshow(velocity.T, cmap='jet')
        plt.colorbar(label='Velocity (km/s)')
        plt.title('Generated Velocity Model')
        plt.xlabel('X (grid points)')
        plt.ylabel('Z (grid points)')
        plt.savefig(os.path.join(self.output_dir, 'test_generated_model.png'))
        plt.close()
        
    def test_model_training(self):
        """测试模型训练功能"""
        # 创建模拟训练数据
        n_samples = 5
        n_rays = 10
        train_data = []
        
        for _ in range(n_samples):
            travel_times = np.random.rand(n_rays)
            ray_paths = [np.array([[0, 0], [self.nx-1, self.nz-1]]) for _ in range(n_rays)]
            source_positions = np.array([[0, 0] for _ in range(n_rays)])
            receiver_positions = np.array([[self.nx-1, self.nz-1] for _ in range(n_rays)])
            velocity = np.random.rand(self.nx, self.nz)
            
            train_data.append((travel_times, ray_paths, source_positions, receiver_positions, velocity))
            
        # 记录训练损失
        train_losses = []
        
        # 测试训练过程
        try:
            self.generator.train(
                train_data=train_data,
                n_epochs=2,
                batch_size=2,
                learning_rate=1e-4,
                callback=lambda epoch, loss: train_losses.append(loss)
            )
            training_successful = True
            
            # 可视化训练损失
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, 'b-o')
            plt.title('Training Loss History')
            plt.xlabel('Batch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, 'test_training_loss.png'))
            plt.close()
            
        except Exception as e:
            training_successful = False
            print(f"训练过程出现错误: {str(e)}")
            
        self.assertTrue(training_successful)
        
    def tearDown(self):
        """测试完成后的清理工作"""
        plt.close('all')  # 关闭所有图形窗口

if __name__ == '__main__':
    unittest.main() 