"""
test_rayinvr_with_actual_files.py - 使用实际的 r.in、v.in、tx.in 文件测试射线追踪功能

测试 zplotpy 中的理论走时计算功能，使用实际的文件进行验证
"""

import sys
from pathlib import Path
import tempfile
import shutil

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 兼容相对导入和绝对导入
try:
    from .theoretical_traveltime import TheoreticalTravelTimeCalculator, ModelLoader
    from .data_loader import DataLoader
    from .pick_manager import PickManager
except ImportError:
    from theoretical_traveltime import TheoreticalTravelTimeCalculator, ModelLoader
    from data_loader import DataLoader
    from pick_manager import PickManager


def test_rayinvr_with_actual_files():
    """使用实际的 r.in、v.in、tx.in 文件测试射线追踪功能"""
    print("\n" + "=" * 70)
    print("使用实际文件测试 RAYINVR 射线追踪功能")
    print("=" * 70)
    
    # 查找实际文件
    zplotpy_dir = Path(__file__).parent
    v_in_path = zplotpy_dir / 'v.in'
    r_in_path = zplotpy_dir / 'r.in'
    tx_in_path = zplotpy_dir / 'tx.in'
    
    # 检查文件是否存在
    missing_files = []
    if not v_in_path.exists():
        missing_files.append(f"v.in: {v_in_path}")
    if not r_in_path.exists():
        missing_files.append(f"r.in: {r_in_path}")
    if not tx_in_path.exists():
        missing_files.append(f"tx.in: {tx_in_path}")
    
    if missing_files:
        print("\n错误: 以下文件不存在:")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\n请确保这些文件存在于: {zplotpy_dir}")
        return False
    
    print(f"\n找到实际文件:")
    print(f"  v.in: {v_in_path}")
    print(f"  r.in: {r_in_path}")
    print(f"  tx.in: {tx_in_path}")
    
    # 创建临时工作目录
    temp_dir = tempfile.mkdtemp(prefix='test_rayinvr_')
    print(f"\n临时工作目录: {temp_dir}")
    
    try:
        # 测试1: 验证模型文件格式
        print("\n" + "=" * 70)
        print("测试 1: 验证模型文件格式")
        print("=" * 70)
        
        loader = ModelLoader()
        is_vin = loader.is_vin_format(str(v_in_path))
        print(f"v.in 文件格式检测: {'v.in 格式' if is_vin else '非 v.in 格式'}")
        
        if not is_vin:
            print("  警告: 文件可能不是 v.in 格式，但继续尝试加载...")
        
        success = loader.load_model(str(v_in_path))
        if not success:
            print("  错误: 模型加载失败")
            return False
        
        model_info = loader.get_model_info()
        print(f"  模型类型: {model_info['model_type']}")
        if model_info.get('n_layers'):
            print(f"  层数: {model_info['n_layers']}")
        print(f"  ✓ 模型加载成功")
        
        # 测试2: 复制实际文件到临时目录
        print("\n" + "=" * 70)
        print("测试 2: 复制实际文件到临时目录")
        print("=" * 70)
        
        import shutil
        temp_v_in = Path(temp_dir) / 'v.in'
        temp_r_in = Path(temp_dir) / 'r.in'
        temp_tx_in = Path(temp_dir) / 'tx.in'
        
        shutil.copy2(v_in_path, temp_v_in)
        shutil.copy2(r_in_path, temp_r_in)
        shutil.copy2(tx_in_path, temp_tx_in)
        
        print(f"  ✓ 文件已复制到临时目录")
        print(f"    {temp_v_in}")
        print(f"    {temp_r_in}")
        print(f"    {temp_tx_in}")
        
        # 验证复制的文件内容（读取前几行）
        print(f"\n  验证复制的 r.in 文件内容（前10行）:")
        with open(temp_r_in, 'r') as f:
            for i, line in enumerate(f.readlines()[:10], 1):
                print(f"    {i}: {line.rstrip()}")
        
        # 检查 r.in 文件中的关键参数
        print(f"\n  检查 r.in 文件中的关键参数:")
        with open(temp_r_in, 'r') as f:
            content = f.read()
            # 检查 imodf 参数
            if 'imodf' in content.lower():
                import re
                match = re.search(r'imodf\s*=\s*(\d+)', content, re.IGNORECASE)
                if match:
                    imodf_value = int(match.group(1))
                    if imodf_value == 1:
                        print(f"    ✓ imodf={imodf_value} (正确，将从 v.in 读取速度模型)")
                    else:
                        print(f"    ⚠ imodf={imodf_value} (应为1才能从 v.in 读取)")
                else:
                    print(f"    ⚠ 找到 imodf 但无法解析值")
            else:
                print(f"    ✗ 未找到 imodf 参数（RAYINVR 将无法从 v.in 读取速度模型）")
            
            # 检查 vfile 参数
            if 'vfile' in content.lower():
                print(f"    ✓ 找到 vfile 参数")
            else:
                print(f"    ⚠ 未找到 vfile 参数（可选）")
        
        # 保存原始 r.in 文件的大小和修改时间
        original_r_in_size = temp_r_in.stat().st_size
        original_r_in_mtime = temp_r_in.stat().st_mtime
        
        # 测试3: 使用实际文件运行 RAYINVR
        print("\n" + "=" * 70)
        print("测试 3: 使用实际文件运行 RAYINVR")
        print("=" * 70)
        
        # 重要：不提供 data_loader，这样即使 auto_generate_inputs=True 也不会生成文件
        # 同时设置 working_dir，确保使用临时目录中的文件
        calculator = TheoreticalTravelTimeCalculator(
            model_file_path=str(temp_v_in),
            working_dir=temp_dir,
            data_loader=None,  # 不提供 data_loader，防止自动生成
            pick_manager=None
        )
        
        # 验证工作目录中的文件
        print(f"\n  验证工作目录中的文件:")
        work_r_in = Path(temp_dir) / 'r.in'
        work_tx_in = Path(temp_dir) / 'tx.in'
        if work_r_in.exists():
            print(f"    r.in 存在: {work_r_in.stat().st_size} 字节")
        else:
            print(f"    r.in 不存在")
        if work_tx_in.exists():
            print(f"    tx.in 存在: {work_tx_in.stat().st_size} 字节")
        else:
            print(f"    tx.in 不存在")
        
        # 检查模型是否加载成功
        model_info = calculator.get_model_info()
        if not model_info['has_model']:
            print("  错误: 计算器模型加载失败")
            return False
        
        print(f"  模型文件: {model_info['model_file']}")
        print(f"  模型类型: {model_info['model_type']}")
        
        # 运行 RAYINVR 前再次验证 r.in 文件
        print(f"\n  运行 RAYINVR 前验证 r.in 文件:")
        print(f"    文件大小: {temp_r_in.stat().st_size} 字节")
        if temp_r_in.stat().st_size != original_r_in_size:
            print(f"    ⚠ 警告: 文件大小已改变！")
        
        # 运行 RAYINVR（使用现有的 r.in 和 tx.in 文件）
        print("\n  开始运行 RAYINVR...")
        print("  （这可能需要一些时间，请耐心等待...）")
        
        try:
            # 不使用自动生成，直接使用现有的文件
            success = calculator.calculate_travel_times(
                auto_generate_inputs=False  # 使用现有的 r.in 和 tx.in
            )
            
            # 运行后再次检查 r.in 文件
            print(f"\n  RAYINVR 运行后检查 r.in 文件:")
            print(f"    文件大小: {temp_r_in.stat().st_size} 字节")
            if temp_r_in.stat().st_size != original_r_in_size:
                print(f"    ⚠ 警告: 文件大小已改变！可能被自动生成覆盖")
                print(f"    原始大小: {original_r_in_size} 字节")
                print(f"    当前大小: {temp_r_in.stat().st_size} 字节")
                print(f"\n    当前 r.in 文件内容（前5行）:")
                with open(temp_r_in, 'r') as f:
                    for i, line in enumerate(f.readlines()[:5], 1):
                        print(f"      {i}: {line.rstrip()}")
            else:
                print(f"    ✓ 文件未被修改")
            
            if success:
                print(f"  ✓ RAYINVR 运行成功")
            else:
                print(f"  ✗ RAYINVR 运行失败")
                return False
                
        except Exception as e:
            print(f"  ✗ RAYINVR 运行出错: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试4: 检查输出文件
        print("\n" + "=" * 70)
        print("测试 4: 检查 RAYINVR 输出文件")
        print("=" * 70)
        
        output_files = {
            'fort.10': 'r.in 输入文件（重定向）',
            'fort.11': 'v.in 输入文件（重定向）',
            'fort.12': 'tx.in 输入文件（重定向）',
            'fort.13': '输出文件',
            'fort.14': '输出文件',
            'fort.15': '输出文件',
            'fort.16': '输出文件',
        }
        
        found_files = []
        for filename, description in output_files.items():
            file_path = Path(temp_dir) / filename
            if file_path.exists():
                size = file_path.stat().st_size
                found_files.append((filename, description, size))
                print(f"  ✓ {filename} ({description}): {size} 字节")
            else:
                print(f"  - {filename} ({description}): 不存在")
        
        if not found_files:
            print("  警告: 没有找到任何输出文件")
        else:
            print(f"\n  找到 {len(found_files)} 个输出文件")
        
        # 测试5: 获取理论走时数据
        print("\n" + "=" * 70)
        print("测试 5: 获取理论走时数据")
        print("=" * 70)
        
        try:
            travel_time_data = calculator.get_theoretical_times()
            if travel_time_data:
                distances = travel_time_data['distance']
                times = travel_time_data['time']
                print(f"  ✓ 成功获取理论走时数据")
                print(f"    数据点数: {len(distances)}")
                if len(distances) > 0:
                    print(f"    距离范围: [{min(distances):.2f}, {max(distances):.2f}] km")
                    print(f"    走时范围: [{min(times):.2f}, {max(times):.2f}] s")
                    print(f"\n    前5个数据点:")
                    for i in range(min(5, len(distances))):
                        print(f"      距离={distances[i]:.2f} km, 走时={times[i]:.3f} s")
            else:
                print(f"  ⚠ 未获取到理论走时数据（可能正常，取决于 RAYINVR 配置）")
        except Exception as e:
            print(f"  ✗ 获取理论走时数据时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试6: 读取观测走时数据（从 tx.in）
        print("\n" + "=" * 70)
        print("测试 6: 读取观测走时数据（从 tx.in）")
        print("=" * 70)
        
        try:
            observed_data = calculator.read_tx_in_file(str(temp_tx_in))
            if observed_data:
                print(f"  ✓ 成功读取 tx.in 文件")
                print(f"    炮点组数: {len(observed_data['shots'])}")
                print(f"    总观测点数: {observed_data['total_observations']}")
                
                if observed_data['shots']:
                    first_shot = observed_data['shots'][0]
                    print(f"    第一个炮点位置: {first_shot['shot_position']:.3f} km")
                    print(f"    第一个炮点的观测点数: {len(first_shot['observations'])}")
                    
                    if first_shot['observations']:
                        first_obs = first_shot['observations'][0]
                        print(f"    第一个观测点:")
                        print(f"      距离={first_obs['x']:.3f} km")
                        print(f"      走时={first_obs['t']:.3f} s")
                        print(f"      误差={first_obs['u']:.3f} s")
                        print(f"      震相={first_obs['phase']}")
            else:
                print(f"  ✗ 读取 tx.in 文件失败")
        except Exception as e:
            print(f"  ✗ 读取 tx.in 文件时出错: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试7: 对比理论走时和观测走时
        print("\n" + "=" * 70)
        print("测试 7: 对比理论走时和观测走时")
        print("=" * 70)
        
        try:
            travel_time_data = calculator.get_theoretical_times()
            observed_data = calculator.read_tx_in_file(str(temp_tx_in))
            
            if travel_time_data and observed_data:
                distances_tt = travel_time_data['distance']
                times_tt = travel_time_data['time']
                
                # 提取观测走时
                obs_distances = []
                obs_times = []
                for shot in observed_data['shots']:
                    for obs in shot['observations']:
                        obs_distances.append(obs['x'])
                        obs_times.append(obs['t'])
                
                if obs_distances:
                    print(f"  理论走时点数: {len(distances_tt)}")
                    print(f"  观测走时点数: {len(obs_distances)}")
                    print(f"\n  观测走时距离范围: [{min(obs_distances):.2f}, {max(obs_distances):.2f}] km")
                    print(f"  观测走时范围: [{min(obs_times):.2f}, {max(obs_times):.2f}] s")
                    
                    # 简单的对比（如果有重叠的距离范围）
                    if distances_tt and obs_distances:
                        tt_min_dist = min(distances_tt)
                        tt_max_dist = max(distances_tt)
                        obs_min_dist = min(obs_distances)
                        obs_max_dist = max(obs_distances)
                        
                        overlap_min = max(tt_min_dist, obs_min_dist)
                        overlap_max = min(tt_max_dist, obs_max_dist)
                        
                        if overlap_min < overlap_max:
                            print(f"\n  距离范围重叠: [{overlap_min:.2f}, {overlap_max:.2f}] km")
                            print(f"  ✓ 可以进行对比分析")
                        else:
                            print(f"\n  ⚠ 距离范围无重叠")
                            print(f"     理论: [{tt_min_dist:.2f}, {tt_max_dist:.2f}] km")
                            print(f"     观测: [{obs_min_dist:.2f}, {obs_max_dist:.2f}] km")
                else:
                    print(f"  ⚠ 没有观测走时数据")
            else:
                print(f"  ⚠ 缺少理论或观测走时数据，无法对比")
        except Exception as e:
            print(f"  ✗ 对比分析时出错: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "=" * 70)
        print("测试完成")
        print("=" * 70)
        print(f"\n临时文件保留在: {temp_dir}")
        print("可以检查以下文件:")
        print(f"  - {temp_v_in}")
        print(f"  - {temp_r_in}")
        print(f"  - {temp_tx_in}")
        print(f"  - {temp_dir}/fort.* (RAYINVR 输出文件)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 保留临时文件以供检查
        print(f"\n注意: 临时文件已保留在 {temp_dir}，可以手动删除")


if __name__ == '__main__':
    success = test_rayinvr_with_actual_files()
    sys.exit(0 if success else 1)
