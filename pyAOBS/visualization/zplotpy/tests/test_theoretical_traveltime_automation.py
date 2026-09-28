"""
test_theoretical_traveltime_automation.py - 理论走时计算自动化功能测试

测试自动生成 r.in 和 tx.in 文件的功能
"""

import sys
from pathlib import Path
import numpy as np
import tempfile
import shutil

# 配置选项：是否清理临时文件
# 设置为 False 可以保留临时文件以供检查
CLEANUP_TEMP_FILES = False

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 兼容相对导入和绝对导入
try:
    from .theoretical_traveltime import TheoreticalTravelTimeCalculator
    from .data_loader import DataLoader, TraceHeader
    from .pick_manager import PickManager
except ImportError:
    from theoretical_traveltime import TheoreticalTravelTimeCalculator
    from data_loader import DataLoader, TraceHeader
    from pick_manager import PickManager


def create_mock_data_loader():
    """创建模拟的 DataLoader 用于测试"""
    data_loader = DataLoader()
    
    # 创建模拟的道头数据
    trace_headers = []
    for i in range(40):
        # 创建模拟道头
        trace_header = TraceHeader(
            ishoti=1,
            itsn=i + 1,
            ireci=i + 1,
            itypei=1,  # 垂直分量
            iflagi=1,  # 有效
            igaini=1,
            offsti=float((i + 1) * 1.0),  # 偏移距 1-40 km
            azi=0.0,
            texact=0.0,
            slat=30.0,
            slong=120.0,
            selev=0.0,
            swdepth=0.0,
            rlat=30.0 + i * 0.01,
            rlong=120.0 + i * 0.01,
            relev=0.0,
            sxutm=0.0,  # 炮点UTM X坐标（米）
            syutm=0.0,
            sz=0.0,  # 炮点深度（km）
            rxutm=float((i + 1) * 1000.0),  # 接收点UTM X坐标（米）
            ryutm=0.0,
            rz=0.0,
            picks=[0.0] * 2,  # 2个拾取字
            picks_from_header=False
        )
        trace_headers.append(trace_header)
    
    data_loader.trace_headers = trace_headers
    
    # 创建模拟的记录文件数据
    try:
        from .data_loader import RecordInfo
    except ImportError:
        from data_loader import RecordInfo
    records = [
        RecordInfo(
            ishnum=1,
            xmod=0.0,  # 模型位置 X（km）
            ymod=0.0,  # 模型位置 Y（km）
            az=0.0,
            title="Test Record"
        )
    ]
    data_loader.records = records
    
    return data_loader


def create_mock_pick_manager():
    """创建模拟的 PickManager 用于测试"""
    pick_manager = PickManager(npick=2)
    
    # 创建模拟的拾取数据
    # 模拟走时曲线：t = sqrt(t0^2 + (x/v)^2)
    t0 = 2.0  # 零偏移走时（秒）
    v = 4.0   # 速度（km/s）
    
    for trace_idx in range(40):
        offset_km = (trace_idx + 1) * 1.0
        theoretical_time = np.sqrt(t0**2 + (offset_km / v)**2)
        # 添加一些随机误差
        np.random.seed(1000 + trace_idx)
        pick_error = np.random.normal(0, 0.03)  # 30ms的标准误差
        pick_time = theoretical_time + pick_error
        
        # 添加拾取（使用第1个拾取字）
        pick_manager.add_pick(trace_idx, pick_time, pick_word=1)
    
    return pick_manager


def test_generate_r_in_file():
    """测试自动生成 r.in 文件"""
    print("=" * 70)
    print("测试 1: 自动生成 r.in 文件")
    print("=" * 70)
    
    # 创建临时工作目录
    temp_dir = tempfile.mkdtemp(prefix='test_tt_automation_')
    print(f"\n工作目录: {temp_dir}")
    
    try:
        # 创建模拟的 DataLoader
        data_loader = create_mock_data_loader()
        
        # 创建计算器（不提供模型文件，只测试生成 r.in）
        calculator = TheoreticalTravelTimeCalculator(
            model_file_path=None,
            working_dir=temp_dir,
            data_loader=data_loader,
            pick_manager=None
        )
        
        # 测试1: 使用默认参数生成 r.in
        print("\n1.1 测试使用默认参数生成 r.in...")
        try:
            r_in_path = calculator.generate_r_in_from_data()
            print(f"   ✓ r.in 文件已生成: {r_in_path}")
            
            # 读取并验证文件内容
            with open(r_in_path, 'r') as f:
                content = f.read()
                print(f"   文件内容预览:")
                for line in content.split('\n')[:5]:
                    if line.strip():
                        print(f"     {line}")
            
            # 验证文件格式
            assert '&pltpar' in content, "缺少 &pltpar 参数块"
            assert '&axepar' in content, "缺少 &axepar 参数块"
            assert '&trapar' in content, "缺少 &trapar 参数块"
            assert 'ray=' in content, "缺少 ray 参数"
            assert 'xshot=' in content, "缺少 xshot 参数"
            
            # 验证关键参数：imodf=1（必须，才能从 v.in 读取速度模型）
            import re
            imodf_match = re.search(r'imodf\s*=\s*(\d+)', content, re.IGNORECASE)
            if imodf_match:
                imodf_value = int(imodf_match.group(1))
                assert imodf_value == 1, f"imodf 应为 1（当前为 {imodf_value}），否则无法从 v.in 读取速度模型"
                print(f"   ✓ imodf=1 参数正确（将从 v.in 读取速度模型）")
            else:
                assert False, "缺少 imodf 参数，RAYINVR 将无法从 v.in 读取速度模型"
            
            # 验证 vfile 参数（可选但推荐）
            if 'vfile' in content.lower():
                print(f"   ✓ vfile 参数已设置")
            else:
                print(f"   ⚠ vfile 参数未设置（可选）")
            
            print(f"   ✓ 文件格式验证通过")
            
        except Exception as e:
            print(f"   ✗ 生成 r.in 文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试2: 使用用户指定参数生成 r.in
        print("\n1.2 测试使用用户指定参数生成 r.in...")
        try:
            shot_position = (5.0, 0.5)  # 用户指定的炮点位置
            ray_params = {
                'ray': [1.2, 2.1],  # 多个震相
                'nray': 10,
                'xmin': 0.0,
                'xmax': 50.0,
                'zmin': 0.0,
                'zmax': 30.0
            }
            
            r_in_path2 = calculator.generate_r_in_from_data(
                shot_position=shot_position,
                ray_params=ray_params
            )
            print(f"   ✓ r.in 文件已生成（用户参数）: {r_in_path2}")
            
            # 验证参数是否正确写入
            with open(r_in_path2, 'r') as f:
                content2 = f.read()
                assert f'xshot={shot_position[0]:.2f}' in content2, "炮点X坐标不正确"
                assert f'zshot={shot_position[1]:.2f}' in content2, "炮点Z坐标不正确"
                assert 'ray=1.2, 2.1' in content2 or 'ray=1.20, 2.10' in content2, "ray参数不正确"
                assert 'nray=10' in content2, "nray参数不正确"
                assert 'xmin=0.0' in content2 or 'xmin=0' in content2, "xmin参数不正确"
                assert 'xmax=50.0' in content2 or 'xmax=50' in content2, "xmax参数不正确"
            
            print(f"   ✓ 用户参数验证通过")
            
        except Exception as e:
            print(f"   ✗ 使用用户参数生成 r.in 文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试3: 测试 ray 参数的不同格式
        print("\n1.3 测试 ray 参数的不同格式...")
        try:
            # 单个值
            ray_params_single = {'ray': 1.2}
            r_in_path3 = calculator.generate_r_in_from_data(ray_params=ray_params_single)
            with open(r_in_path3, 'r') as f:
                content3 = f.read()
                assert 'ray=1.2' in content3 or 'ray=1.20' in content3, "单个ray值格式错误"
            print(f"   ✓ 单个 ray 值格式正确")
            
            # 多个值
            ray_params_multi = {'ray': [1.2, 2.1, 3.1]}
            r_in_path4 = calculator.generate_r_in_from_data(ray_params=ray_params_multi)
            with open(r_in_path4, 'r') as f:
                content4 = f.read()
                assert 'ray=1.2, 2.1, 3.1' in content4 or 'ray=1.20, 2.10, 3.10' in content4, "多个ray值格式错误"
            print(f"   ✓ 多个 ray 值格式正确")
            
        except Exception as e:
            print(f"   ✗ ray 参数格式测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n{'='*70}")
        print("✓ 测试 1 通过：自动生成 r.in 文件功能正常")
        print(f"{'='*70}\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试 1 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时目录（如果配置允许）
        if CLEANUP_TEMP_FILES:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
                print(f"已清理临时目录: {temp_dir}")
        else:
            print(f"保留临时目录以供检查: {temp_dir}")


def test_generate_tx_in_file():
    """测试自动生成 tx.in 文件"""
    print("=" * 70)
    print("测试 2: 自动生成 tx.in 文件（符合 RAYINVR 格式）")
    print("=" * 70)
    
    # 创建临时工作目录
    temp_dir = tempfile.mkdtemp(prefix='test_tt_automation_')
    print(f"\n工作目录: {temp_dir}")
    
    try:
        # 创建模拟的 DataLoader 和 PickManager
        data_loader = create_mock_data_loader()
        pick_manager = create_mock_pick_manager()
        
        # 创建计算器
        calculator = TheoreticalTravelTimeCalculator(
            model_file_path=None,
            working_dir=temp_dir,
            data_loader=data_loader,
            pick_manager=pick_manager
        )
        
        # 测试1: 使用默认参数生成 tx.in
        print("\n2.1 测试使用默认参数生成 tx.in...")
        try:
            tx_in_path = calculator.generate_tx_in_from_picks()
            
            if tx_in_path:
                print(f"   ✓ tx.in 文件已生成: {tx_in_path}")
                
                # 读取并验证文件内容
                with open(tx_in_path, 'r') as f:
                    lines = f.readlines()
                    print(f"   文件总行数: {len(lines)}")
                    print(f"   前5行内容:")
                    for i, line in enumerate(lines[:5]):
                        print(f"     {i+1}: {line.rstrip()}")
                
                # 验证文件格式（固定宽度格式）
                # 注意：RAYINVR 格式要求每行固定40字符（不包括换行符）
                # 使用 rstrip() 只去掉换行符，保留前导和中间的空格
                if len(lines) > 0:
                    first_line = lines[0].rstrip('\n\r')
                    # 验证第一行长度（应为40字符，不包括换行符）
                    assert len(first_line) == 40, f"第一行长度不正确（应为40字符，实际为{len(first_line)}字符）"
                    # 验证格式：xshot -1.000 0.000 0
                    parts = [first_line[i:i+10] for i in range(0, 40, 10)]
                    assert float(parts[1].strip()) == -1.0, "炮点组标记格式错误"
                    assert int(parts[3].strip()) == 0, "炮点组标记格式错误"
                    print(f"   ✓ 炮点组开始标记格式正确（长度：{len(first_line)}字符）")
                
                # 验证最后一行是文件结束标记
                if len(lines) > 1:
                    last_line = lines[-1].rstrip('\n\r')
                    if last_line:
                        assert len(last_line) == 40, f"最后一行长度不正确（应为40字符，实际为{len(last_line)}字符）"
                        parts = [last_line[i:i+10] for i in range(0, 40, 10)]
                        assert int(parts[3].strip()) == -1, "文件结束标记格式错误"
                        print(f"   ✓ 文件结束标记格式正确（长度：{len(last_line)}字符）")
                
                # 验证观测走时点格式
                if len(lines) > 2:
                    obs_line = lines[1].rstrip('\n\r')
                    if obs_line:
                        assert len(obs_line) == 40, f"观测走时点行长度不正确（应为40字符，实际为{len(obs_line)}字符）"
                        parts = [obs_line[i:i+10] for i in range(0, 40, 10)]
                        # 验证每列都是10位宽度
                        assert len(parts) == 4, "观测走时点格式错误（应为4列）"
                        # 验证第4列（震相标识）是正整数
                        phase_id = int(parts[3].strip())
                        assert phase_id > 0, "震相标识应为正整数"
                        print(f"   ✓ 观测走时点格式正确（长度：{len(obs_line)}字符）")
                
            else:
                print(f"   ✗ 生成 tx.in 文件失败（返回 None）")
                return False
                
        except Exception as e:
            print(f"   ✗ 生成 tx.in 文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试2: 使用用户指定参数生成 tx.in
        print("\n2.2 测试使用用户指定参数生成 tx.in...")
        try:
            shot_position = (10.0, 0.5)
            phase_id = 2
            uncertainty = 0.100
            
            tx_in_path2 = calculator.generate_tx_in_from_picks(
                pick_word=1,
                shot_position=shot_position,
                phase_id=phase_id,
                uncertainty=uncertainty
            )
            
            if tx_in_path2:
                print(f"   ✓ tx.in 文件已生成（用户参数）: {tx_in_path2}")
                
                # 验证参数是否正确写入
                with open(tx_in_path2, 'r') as f:
                    lines2 = f.readlines()
                    if len(lines2) > 0:
                        first_line = lines2[0].rstrip('\n\r')  # 只去掉换行符
                        assert len(first_line) == 40, f"第一行长度不正确（应为40字符，实际为{len(first_line)}字符）"
                        parts = [first_line[i:i+10] for i in range(0, 40, 10)]
                        shot_x = float(parts[0].strip())
                        assert abs(shot_x - shot_position[0]) < 0.001, "炮点X坐标不正确"
                        print(f"   ✓ 炮点位置验证通过: {shot_x:.3f}")
                    
                    # 验证震相标识和误差
                    if len(lines2) > 1:
                        obs_line = lines2[1].rstrip('\n\r')  # 只去掉换行符
                        if obs_line and len(obs_line) >= 40:
                            parts = [obs_line[i:i+10] for i in range(0, 40, 10)]
                            obs_phase = int(parts[3].strip())
                            obs_uncertainty = float(parts[2].strip())
                            assert obs_phase == phase_id, f"震相标识不正确: {obs_phase} != {phase_id}"
                            assert abs(obs_uncertainty - uncertainty) < 0.001, f"误差值不正确: {obs_uncertainty} != {uncertainty}"
                            print(f"   ✓ 震相标识验证通过: {obs_phase}")
                            print(f"   ✓ 误差值验证通过: {obs_uncertainty:.3f}")
                
            else:
                print(f"   ✗ 使用用户参数生成 tx.in 文件失败")
                return False
                
        except Exception as e:
            print(f"   ✗ 使用用户参数生成 tx.in 文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试3: 验证数据排序
        print("\n2.3 测试观测走时点排序...")
        try:
            tx_in_path3 = calculator.generate_tx_in_from_picks()
            if tx_in_path3:
                with open(tx_in_path3, 'r') as f:
                    lines3 = f.readlines()
                
                # 提取观测走时点的距离
                distances = []
                for line in lines3[1:-1]:  # 跳过第一行（炮点标记）和最后一行（结束标记）
                    line = line.rstrip('\n\r')  # 只去掉换行符，保留空格
                    if line and len(line) >= 40:  # 确保行长度足够
                        parts = [line[i:i+10] for i in range(0, 40, 10)]
                        ipf = int(parts[3].strip())
                        if ipf > 0:  # 观测走时点
                            xpf = float(parts[0].strip())
                            distances.append(xpf)
                
                # 验证是否按距离排序
                if len(distances) > 1:
                    is_sorted = all(distances[i] <= distances[i+1] for i in range(len(distances)-1))
                    assert is_sorted, "观测走时点未按距离排序"
                    print(f"   ✓ 观测走时点已按距离排序（共 {len(distances)} 个点）")
                    print(f"     距离范围: [{min(distances):.3f}, {max(distances):.3f}] km")
                
        except Exception as e:
            print(f"   ✗ 数据排序验证失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n{'='*70}")
        print("✓ 测试 2 通过：自动生成 tx.in 文件功能正常")
        print(f"{'='*70}\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试 2 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时目录（如果配置允许）
        if CLEANUP_TEMP_FILES:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
                print(f"已清理临时目录: {temp_dir}")
        else:
            print(f"保留临时目录以供检查: {temp_dir}")


def test_read_tx_in_file():
    """测试读取和验证 tx.in 文件"""
    print("=" * 70)
    print("测试 3: 读取和验证 tx.in 文件")
    print("=" * 70)
    
    # 创建临时工作目录
    temp_dir = tempfile.mkdtemp(prefix='test_tt_automation_')
    print(f"\n工作目录: {temp_dir}")
    
    try:
        # 创建模拟的 DataLoader 和 PickManager
        data_loader = create_mock_data_loader()
        pick_manager = create_mock_pick_manager()
        
        # 创建计算器并生成 tx.in 文件
        calculator = TheoreticalTravelTimeCalculator(
            model_file_path=None,
            working_dir=temp_dir,
            data_loader=data_loader,
            pick_manager=pick_manager
        )
        
        # 生成测试用的 tx.in 文件
        tx_in_path = calculator.generate_tx_in_from_picks()
        if not tx_in_path:
            print("   ✗ 无法生成测试用的 tx.in 文件")
            return False
        
        print(f"\n3.1 测试读取 tx.in 文件...")
        try:
            data = calculator.read_tx_in_file(tx_in_path)
            
            if data:
                print(f"   ✓ 成功读取 tx.in 文件")
                print(f"   炮点组数: {len(data['shots'])}")
                print(f"   总观测点数: {data['total_observations']}")
                
                # 验证数据结构
                assert 'shots' in data, "缺少 'shots' 键"
                assert 'total_observations' in data, "缺少 'total_observations' 键"
                assert len(data['shots']) > 0, "没有炮点组"
                
                # 验证第一个炮点组
                first_shot = data['shots'][0]
                assert 'shot_position' in first_shot, "缺少 'shot_position' 键"
                assert 'observations' in first_shot, "缺少 'observations' 键"
                assert len(first_shot['observations']) > 0, "没有观测点"
                
                print(f"   第一个炮点位置: {first_shot['shot_position']:.3f} km")
                print(f"   第一个炮点的观测点数: {len(first_shot['observations'])}")
                
                # 验证观测点数据结构
                first_obs = first_shot['observations'][0]
                assert 'x' in first_obs, "缺少 'x' 键"
                assert 't' in first_obs, "缺少 't' 键"
                assert 'u' in first_obs, "缺少 'u' 键"
                assert 'phase' in first_obs, "缺少 'phase' 键"
                
                print(f"   第一个观测点: 距离={first_obs['x']:.3f} km, "
                      f"走时={first_obs['t']:.3f} s, "
                      f"误差={first_obs['u']:.3f} s, "
                      f"震相={first_obs['phase']}")
                
            else:
                print(f"   ✗ 读取 tx.in 文件失败（返回 None）")
                return False
                
        except Exception as e:
            print(f"   ✗ 读取 tx.in 文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试2: 验证文件格式
        print("\n3.2 测试验证 tx.in 文件格式...")
        try:
            is_valid, error_msg = calculator.validate_tx_in_file(tx_in_path)
            
            if is_valid:
                print(f"   ✓ 文件格式验证通过")
            else:
                print(f"   ✗ 文件格式验证失败: {error_msg}")
                return False
                
        except Exception as e:
            print(f"   ✗ 文件格式验证失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试3: 测试读取不存在的文件
        print("\n3.3 测试读取不存在的文件...")
        try:
            non_existent_file = Path(temp_dir) / 'non_existent_tx.in'
            data = calculator.read_tx_in_file(str(non_existent_file))
            assert data is None, "应该返回 None"
            print(f"   ✓ 正确处理不存在的文件")
        except Exception as e:
            print(f"   ✗ 处理不存在文件时出错: {e}")
            return False
        
        print(f"\n{'='*70}")
        print("✓ 测试 3 通过：读取和验证 tx.in 文件功能正常")
        print(f"{'='*70}\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试 3 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时目录（如果配置允许）
        if CLEANUP_TEMP_FILES:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
                print(f"已清理临时目录: {temp_dir}")
        else:
            print(f"保留临时目录以供检查: {temp_dir}")


def test_full_automation_workflow():
    """测试完整的自动化工作流程"""
    print("=" * 70)
    print("测试 4: 完整的自动化工作流程（需要实际的 v.in 文件）")
    print("=" * 70)
    
    # 提示用户提供模型文件
    print("\n注意：此测试需要提供实际的 v.in 格式速度模型文件")
    print("如果跳过此测试，不影响其他测试结果\n")
    
    model_file = input("请输入 v.in 文件路径（或按回车跳过）: ").strip()
    
    if not model_file:
        print("   跳过测试（未提供文件路径）")
        return True
    
    # 验证文件路径
    model_path = Path(model_file)
    
    # 如果输入的是目录，尝试在该目录下查找 v.in 文件
    if model_path.is_dir():
        vin_file = model_path / 'v.in'
        if vin_file.exists() and vin_file.is_file():
            print(f"   在目录中找到 v.in 文件: {vin_file}")
            model_file = str(vin_file)
            model_path = vin_file
        else:
            print(f"   ✗ 错误: 输入的是目录，且目录中未找到 v.in 文件")
            print(f"   目录路径: {model_path}")
            return False
    elif not model_path.exists():
        print(f"   ✗ 错误: 文件或目录不存在: {model_path}")
        return False
    elif not model_path.is_file():
        print(f"   ✗ 错误: 输入路径不是文件: {model_path}")
        return False
    
    # 验证文件扩展名（可选，但有助于用户确认）
    if model_path.suffix.lower() not in ['.in', '']:
        print(f"   ⚠ 警告: 文件扩展名不是 .in: {model_path.suffix}")
        print(f"   继续尝试加载...")
    
    model_file = str(model_path)
    
    # 创建临时工作目录
    temp_dir = tempfile.mkdtemp(prefix='test_tt_automation_')
    print(f"\n工作目录: {temp_dir}")
    
    try:
        # 创建模拟的 DataLoader 和 PickManager
        data_loader = create_mock_data_loader()
        pick_manager = create_mock_pick_manager()
        
        # 创建计算器（使用实际的模型文件）
        calculator = TheoreticalTravelTimeCalculator(
            model_file_path=model_file,
            working_dir=temp_dir,
            data_loader=data_loader,
            pick_manager=pick_manager
        )
        
        # 检查模型是否加载成功
        model_info = calculator.get_model_info()
        if not model_info['has_model']:
            print(f"   ✗ 模型加载失败")
            print(f"   文件路径: {model_file}")
            print(f"   提示: 请确保文件是有效的 v.in 格式速度模型文件")
            return False
        
        if model_info['model_type'] != 'vin':
            print(f"   ✗ 模型格式不正确")
            print(f"   文件路径: {model_file}")
            print(f"   检测到的格式: {model_info.get('model_type', '未知')}")
            print(f"   提示: RAYINVR 需要 v.in 格式的速度模型文件")
            return False
        
        print(f"   ✓ 模型加载成功")
        print(f"   文件路径: {model_info['model_file']}")
        print(f"   模型类型: {model_info['model_type']}")
        if 'n_layers' in model_info:
            print(f"   层数: {model_info['n_layers']}")
        
        # 测试完整的自动化流程
        print("\n4.1 测试完整的自动化流程（自动生成 r.in 和 tx.in）...")
        try:
            # 使用用户指定的参数
            shot_position = (0.0, 0.0)
            ray_params = {
                'ray': [1.2],  # 第1层反射
                'nray': 5
            }
            
            # 计算理论走时（自动生成输入文件）
            print("   正在自动生成 r.in 和 tx.in 文件...")
            success = calculator.calculate_travel_times(
                auto_generate_inputs=True,
                shot_position=shot_position,
                ray_params=ray_params,
                use_observed_picks=True,  # 使用观测拾取生成 tx.in
                pick_word=1
            )
            
            if success:
                print(f"   ✓ RAYINVR 计算完成")
                
                # 验证生成的文件
                r_in_path = Path(temp_dir) / 'r.in'
                tx_in_path = Path(temp_dir) / 'tx.in'
                
                if r_in_path.exists():
                    print(f"   ✓ r.in 文件已生成: {r_in_path}")
                    with open(r_in_path, 'r') as f:
                        r_content = f.read()
                        print(f"     文件大小: {len(r_content)} 字节")
                        print(f"     包含 ray 参数: {'ray=' in r_content}")
                
                if tx_in_path.exists():
                    print(f"   ✓ tx.in 文件已生成: {tx_in_path}")
                    with open(tx_in_path, 'r') as f:
                        tx_lines = f.readlines()
                        print(f"     文件行数: {len(tx_lines)}")
                        print(f"     文件大小: {sum(len(line) for line in tx_lines)} 字节")
                
                # 获取理论走时数据
                print("\n4.2 获取理论走时数据...")
                travel_time_data = calculator.get_theoretical_times()
                if travel_time_data:
                    distances = travel_time_data['distance']
                    times = travel_time_data['time']
                    print(f"   ✓ 成功获取理论走时数据")
                    print(f"     数据点数: {len(distances)}")
                    print(f"     距离范围: [{np.min(distances):.2f}, {np.max(distances):.2f}] km")
                    print(f"     走时范围: [{np.min(times):.2f}, {np.max(times):.2f}] s")
                else:
                    print(f"   ⚠ 未获取到理论走时数据（可能正常）")
                
            else:
                print(f"   ✗ RAYINVR 计算失败")
                return False
                
        except Exception as e:
            print(f"   ✗ 自动化流程测试失败: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print(f"\n{'='*70}")
        print("✓ 测试 4 通过：完整的自动化工作流程正常")
        print(f"{'='*70}\n")
        return True
        
    except Exception as e:
        print(f"\n✗ 测试 4 失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 清理临时目录（如果配置允许）
        if CLEANUP_TEMP_FILES:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
                print(f"已清理临时目录: {temp_dir}")
        else:
            print(f"保留临时目录以供检查: {temp_dir}")


def main():
    """主测试函数"""
    print("\n" + "=" * 70)
    print("理论走时计算自动化功能测试")
    print("=" * 70)
    print("\n本测试将验证以下功能:")
    print("1. 自动生成 r.in 文件（最小化参数，支持用户修改）")
    print("2. 自动生成 tx.in 文件（符合 RAYINVR 格式要求）")
    print("3. 读取和验证 tx.in 文件")
    print("4. 完整的自动化工作流程（需要实际的 v.in 文件）")
    print("=" * 70 + "\n")
    
    results = []
    
    try:
        # 测试1: 自动生成 r.in 文件
        result1 = test_generate_r_in_file()
        results.append(("自动生成 r.in 文件", result1))
        
        # 测试2: 自动生成 tx.in 文件
        result2 = test_generate_tx_in_file()
        results.append(("自动生成 tx.in 文件", result2))
        
        # 测试3: 读取和验证 tx.in 文件
        result3 = test_read_tx_in_file()
        results.append(("读取和验证 tx.in 文件", result3))
        
        # 测试4: 完整的自动化工作流程（可选）
        user_input = input("\n是否继续测试完整的自动化工作流程（需要 v.in 文件）？(y/n): ").strip().lower()
        if user_input == 'y':
            result4 = test_full_automation_workflow()
            results.append(("完整的自动化工作流程", result4))
        else:
            print("   跳过测试 4")
        
        # 输出测试总结
        print("\n" + "=" * 70)
        print("测试总结")
        print("=" * 70)
        for test_name, result in results:
            status = "✓ 通过" if result else "✗ 失败"
            print(f"{test_name}: {status}")
        
        all_passed = all(result for _, result in results)
        print("=" * 70)
        if all_passed:
            print("✓ 所有测试通过！")
        else:
            print("✗ 部分测试失败，请检查上述输出")
        print("=" * 70 + "\n")
        
        return all_passed
        
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        return False
    except Exception as e:
        print(f"\n\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
