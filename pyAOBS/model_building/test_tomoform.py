"""
test_tomoform.py - Test cases for tomoform module

This module contains test cases for the velocity model manipulation tools
implemented in tomoform.py.

Author: Haibo Huang
Date: 2025
"""

import unittest
import numpy as np
from pathlib import Path
import os
import tempfile
from .tomoform import SlownessMesh2D, VelocityModelGenerator, plot_velocity_model


class TestSlownessMesh2D(unittest.TestCase):
    """Test cases for SlownessMesh2D class"""
    
    def setUp(self):
        """Set up test cases"""
        # Create a simple test model
        self.nx = 41
        self.nz = 31
        self.v_water = 1.5
        self.v_air = 0.33
        self.mesh = SlownessMesh2D(self.nx, self.nz, self.v_water, self.v_air)
        self.mesh.xpos = np.linspace(0, 40, self.nx)
        self.mesh.zpos = np.linspace(-1, 8, self.nz)  # 从-1km到8km
        self.mesh.topo = np.ones(self.nx) * 0.5  # 海底在0.5km深度
        
        # 设置速度场：考虑空气、水和地下介质
        self.mesh.vgrid = np.ones((self.nx, self.nz)) * 3.0  # 默认地下速度
        for i in range(self.nx):
            for j in range(self.nz):
                if self.mesh.zpos[j] < 0:  # 海平面以上
                    self.mesh.vgrid[i,j] = self.v_air
                elif self.mesh.zpos[j] <= 0 and self.mesh.zpos[j] <= self.mesh.topo[i]:  # 海水层
                    self.mesh.vgrid[i,j] = self.v_water
        
        self.mesh.pgrid = 1.0 / self.mesh.vgrid
        
        # Create test directory if it doesn't exist
        self.test_dir = Path(__file__).parent / "test_output"
        self.test_dir.mkdir(exist_ok=True)
        
    def test_init(self):
        """Test initialization"""
        mesh = SlownessMesh2D(self.nx, self.nz, self.v_water, self.v_air)
        self.assertEqual(mesh.nx, self.nx)
        self.assertEqual(mesh.nz, self.nz)
        self.assertEqual(mesh.v_water, self.v_water)
        self.assertEqual(mesh.v_air, self.v_air)
        
    def test_file_io(self):
        """Test file reading and writing"""
        # Write test model
        test_file = Path(self.test_dir) / "test.smesh"
        self.mesh.to_file(str(test_file))
        
        # Read back and compare
        mesh2 = SlownessMesh2D.from_file(str(test_file))
        
        # Compare coordinates
        np.testing.assert_array_almost_equal(mesh2.xpos, self.mesh.xpos)
        np.testing.assert_array_almost_equal(mesh2.zpos, self.mesh.zpos)
        np.testing.assert_array_almost_equal(mesh2.topo, self.mesh.topo)
        
        # Compare velocity field
        np.testing.assert_array_almost_equal(mesh2.vgrid, self.mesh.vgrid)
        np.testing.assert_array_almost_equal(mesh2.pgrid, self.mesh.pgrid)
        
    def test_smooth(self):
        """Test smoothing operation"""
        # Create a non-uniform velocity field
        x, z = np.meshgrid(self.mesh.xpos, self.mesh.zpos, indexing='ij')
        self.mesh.vgrid = 3.0 + 0.5 * np.sin(2*np.pi*x/20.0)  # Add some horizontal variation
        original_vgrid = self.mesh.vgrid.copy()
        
        # Apply smoothing
        self.mesh.gaussian_smooth(Lh=5.0, Lv=2.0)
        
        # Check that smoothing changed the values
        self.assertFalse(np.array_equal(self.mesh.vgrid, original_vgrid))
        
        # Check that average velocity remains similar
        np.testing.assert_almost_equal(
            np.mean(self.mesh.vgrid),
            np.mean(original_vgrid),
            decimal=2
        )
        
        # Check that smoothing reduced the variance
        self.assertLess(np.var(self.mesh.vgrid), np.var(original_vgrid))
        
    def test_checkerboard(self):
        """Test checkerboard pattern"""
        # 保存原始速度场
        original_vgrid = self.mesh.vgrid.copy()
        
        # 创建地下介质掩码（排除空气和水层）
        x, z = np.meshgrid(self.mesh.xpos, self.mesh.zpos, indexing='ij')
        subsurface_mask = (z > self.mesh.topo[:, np.newaxis])
        
        # 记录原始地下速度的平均值
        v_subsurface = np.mean(original_vgrid[subsurface_mask])
        
        # 添加棋盘格图案
        self.mesh.add_checkerboard(amplitude=5.0, ch=10.0, cv=2.0)
        
        # 检查地下区域的变化
        diff = self.mesh.vgrid - original_vgrid
        max_diff = np.max(np.abs(diff[subsurface_mask]))
        
        # 检查图案是否被添加
        self.assertFalse(np.array_equal(
            self.mesh.vgrid[subsurface_mask], 
            original_vgrid[subsurface_mask]
        ))
        
        # 检查振幅（5% + 小的数值误差）
        self.assertLess(max_diff, v_subsurface * 0.06)
        
        # 检查空气和水层是否保持不变
        np.testing.assert_array_equal(
            self.mesh.vgrid[~subsurface_mask],
            original_vgrid[~subsurface_mask]
        )
        
    def test_anomaly(self):
        """Test rectangular anomaly"""
        original_vgrid = self.mesh.vgrid.copy()
        self.mesh.add_anomaly(
            amplitude=10.0,
            xmin=10.0, xmax=20.0,
            zmin=2.0, zmax=4.0
        )
        
        # Check that anomaly was added
        self.assertFalse(np.array_equal(self.mesh.vgrid, original_vgrid))
        
        # Check that values outside anomaly remain unchanged
        mask = np.zeros_like(self.mesh.vgrid, dtype=bool)
        x, z = np.meshgrid(self.mesh.xpos, self.mesh.zpos, indexing='ij')
        mask[(x >= 10.0) & (x <= 20.0) & (z >= 2.0) & (z <= 4.0)] = True
        np.testing.assert_array_equal(
            self.mesh.vgrid[~mask],
            original_vgrid[~mask]
        )
        
    def test_gaussian(self):
        """Test Gaussian anomaly"""
        original_vgrid = self.mesh.vgrid.copy()
        self.mesh.add_gaussian(
            amplitude=10.0,
            x0=20.0, z0=4.0,
            Lh=5.0, Lv=2.0
        )
        
        # Check that anomaly was added
        self.assertFalse(np.array_equal(self.mesh.vgrid, original_vgrid))
        
        # Check that maximum perturbation is at the center
        x, z = np.meshgrid(self.mesh.xpos, self.mesh.zpos, indexing='ij')
        center_idx = np.argmin(np.abs(x - 20.0) + np.abs(z - 4.0))
        diff = np.abs(self.mesh.vgrid - original_vgrid)
        self.assertEqual(np.argmax(diff), center_idx)
        
class TestVelocityModelGenerator(unittest.TestCase):
    """Test cases for VelocityModelGenerator class"""
    
    def test_uniform_gradient(self):
        """Test uniform gradient model generation"""
        mesh = VelocityModelGenerator.uniform_gradient(
            nx=41, nz=31,
            xmax=40.0, zmax=8.0,
            v0=3.0, gradient=0.1
        )
        
        # Check dimensions
        self.assertEqual(mesh.vgrid.shape, (41, 31))
        
        # Check gradient
        z = np.tile(mesh.zpos, (41, 1))
        expected_vgrid = 3.0 + 0.1 * z
        np.testing.assert_array_almost_equal(mesh.vgrid, expected_vgrid)
        
    def test_from_interfaces(self):
        """Test interface-based model generation"""
        # Define two interfaces
        x = np.linspace(0, 40, 41)
        z1 = np.ones_like(x) * 2.0  # Flat interface at z=2
        z2 = np.ones_like(x) * 4.0  # Flat interface at z=4
        
        interfaces = [
            (x, z1, 4.0),  # v=4.0 below first interface
            (x, z2, 5.0)   # v=5.0 below second interface
        ]
        
        mesh = VelocityModelGenerator.from_interfaces(
            interfaces=interfaces,
            nx=41, nz=31,
            xmax=40.0, zmax=8.0
        )
        
        # Check dimensions
        self.assertEqual(mesh.vgrid.shape, (41, 31))
        
        # Check velocities in different layers
        z = np.tile(mesh.zpos, (41, 1))
        self.assertTrue(np.all(mesh.vgrid[:, mesh.zpos < 2.0] == 1.5))  # Water
        self.assertTrue(np.all(mesh.vgrid[:, (mesh.zpos >= 2.0) & (mesh.zpos < 4.0)] == 4.0))
        self.assertTrue(np.all(mesh.vgrid[:, mesh.zpos >= 4.0] == 5.0))

class TestCreateModel(unittest.TestCase):
    """Test model creation functionality"""
    
    def setUp(self):
        """Set up test case"""
        self.test_dir = Path(__file__).parent / 'test_output'
        self.test_dir.mkdir(exist_ok=True)
        self.smesh_file = Path(__file__).parent / 'smesh_vs05_S.dat'
        self.test_file = Path(self.test_dir) / "test_file.smesh"
        mesh2 = SlownessMesh2D.from_file(str(self.smesh_file))
        mesh2.to_file(str(self.test_file))
        
    def test_create_model_with_plot(self):
        """Test model creation with plotting"""
        mesh, plot_file = plot_velocity_model(
            str(self.smesh_file),
            output_dir=str(self.test_dir),
            plot_interfaces=True,
            interface_color='black',
            interface_linewidth=1.0,
            interface_linestyle='-',
            plot_contours=True,
            plot_region=[0, 410, 0, 30],
            contour_interval=0.5,
            contour_levels=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            cmap='jet'
        )

        self.assertIsInstance(mesh, SlownessMesh2D)
        self.assertIsNotNone(plot_file)
        plot_path = Path(plot_file)
        self.assertTrue(plot_path.exists())
        self.assertEqual(plot_path.name, 'velocity_model.png')
        
    def test_create_model_invalid_file(self):
        """Test model creation with invalid file"""
        with self.assertRaises(FileNotFoundError):
            plot_velocity_model(
                'nonexistent.smesh',
                output_dir=str(self.test_dir)
            )


if __name__ == '__main__':
    unittest.main() 