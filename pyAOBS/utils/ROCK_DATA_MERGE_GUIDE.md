# Rock 数据并库模板

用于把外部岩石物性数据（Vp/Vs/密度/温压等）合并到 `utils/rockdata` 下的数据库文件。

## 1) 准备外部数据

复制模板文件并填充数据：

- `utils/rockdata/external_rock_data_template.csv`

模板列说明：

- `rock_type` 必填，岩石名称
- `density` 必填，密度值
- `density_unit` 必填，支持 `kg/m3` 或 `g/cm3`
- `pressure` 必填，压力值
- `pressure_unit` 建议填写，支持 `MPa` 或 `GPa`（并库时统一转成 MPa）
- `vp` / `vs` 必填，单位 `km/s`
- 可选扩展特征：`felsic_or_mafic`、`rock_facies`、`sio2_wt`
- `pressure` / `temperature` 必填（建议统一到与你现有库一致条件，例如 200 MPa / 25 C）
- `source` / `method` 必填，用于溯源
- 其他列（`abbrev`、`岩石属性`、`dvp`、`dvs`、`det_den`）可选

## 2) 来源数据一键映射（推荐）

如果你的数据来自 P3/USGS/EarthChem，先做字段映射（支持 `csv/xlsx/xlsm`）：

```bash
python -m pyAOBS.utils.prepare_external_rock_data --source-csv "d:/path/to/source.csv" --preset p3 --out "d:/python-learn/pyAOBS/pyAOBS/utils/rockdata/external_rock_data_template.csv"
```

如果是 Excel 且需要指定工作表：

```bash
python -m pyAOBS.utils.prepare_external_rock_data --source-csv "d:/path/to/source.xlsx" --preset p3 --sheet "Sheet1" --out "d:/python-learn/pyAOBS/pyAOBS/utils/rockdata/external_rock_data_template.csv"
```

可选预置：

- `p3`
- `p3_518`（518 Appendix A 专用解译器，支持按 `Pressure, Gpa` 多列展开样本）
- `usgs`
- `earthchem`

518 专用推荐命令：

```bash
python -m pyAOBS.utils.prepare_external_rock_data --source-csv "pyAOBS/utils/rockdata/518-1_AppendixAseismicvelocitydata.xlsx" --preset p3_518 --sheet "Data Master"
```

如需只保留指定压力窗口（GPa），例如 `0.1-0.6`：

```bash
python -m pyAOBS.utils.prepare_external_rock_data --source-csv "pyAOBS/utils/rockdata/518-1_AppendixAseismicvelocitydata.xlsx" --preset p3_518 --sheet "Data Master" --min-pressure-gpa 0.1 --max-pressure-gpa 0.6
```

如需转换后直接输出 `source` 统计：

```bash
python -m pyAOBS.utils.prepare_external_rock_data --source-csv "pyAOBS/utils/rockdata/518-1_AppendixAseismicvelocitydata.xlsx" --preset p3_518 --sheet "Data Master" --report-sources --report-top 30
```

映射规则文件：

- `utils/rock_source_field_mappings.json`

你也可以直接编辑该 JSON，把你实际下载表格里的列名加到对应字段别名列表里。

## 3) 执行并库

在项目根目录执行：

```bash
python -m pyAOBS.utils.merge_rock_database --incoming "d:/python-learn/pyAOBS/pyAOBS/utils/rockdata/external_rock_data_template.csv"
```

默认会优先更新 `utils/rockdata/rocks_merged.csv`（若存在）；
否则依次使用 `utils/rockdata/rocks.csv`、`utils/rocks.csv` 作为基库。
如需先预览到新文件：

```bash
python -m pyAOBS.utils.merge_rock_database --incoming "d:/python-learn/pyAOBS/pyAOBS/utils/rockdata/external_rock_data_template.csv" --out "d:/python-learn/pyAOBS/pyAOBS/utils/rockdata/rocks_merged_preview.csv"
```

## 4) 脚本做了什么

- 自动把密度统一到 `kg/m3`
  - `g/cm3` -> `kg/m3`（乘以 1000）
  - 未提供单位时按数值推断（`<20` 视为 `g/cm3`）
- 自动把压力统一到 `MPa`
  - `GPa` -> `MPa`（乘以 1000）
  - 未提供单位时按数值推断（`<=5` 视为 `GPa`）
- 仅追加新记录（去重）
  - 去重键：`rock_type + density + vp + vs + pressure + temperature + source + method`
- 跳过不完整行（会统计 `skipped_invalid`）

## 5) 建议工作流

- 先用 `prepare_external_rock_data` 统一字段
- 再用 `merge_rock_database` 合并入库
- 对合并命令先 `--out` 预览结果
- 抽查几条岩石（单位、温压条件、source/method）
- 确认无误后覆盖到 `utils/rockdata/rocks_merged.csv`

## 6) 数据库健康检查（新增）

并库后可快速检查样本完整性和来源分布：

```bash
python -m pyAOBS.utils.rock_database_health_check --db "pyAOBS/utils/rockdata/rocks_merged.csv" --top 20
```

## 7) 一键脚本（新增）

已提供两个一键脚本（放在 `utils/rockdata`）：

- Windows PowerShell: `build_rocks_merged.ps1`
- Linux/WSL Bash: `build_rocks_merged.sh`

PowerShell 使用示例（项目根目录执行）：

```powershell
.\pyAOBS\utils\rockdata\build_rocks_merged.ps1
```

Bash 使用示例（项目根目录执行）：

```bash
bash pyAOBS/utils/rockdata/build_rocks_merged.sh
```

## 8) 中文名回填（新增）

如果 `rocks_merged.csv` 中 `岩石属性` 有缺失，可自动回填：

```bash
python -m pyAOBS.utils.fill_rock_cn_names --input "pyAOBS/utils/rockdata/rocks_merged.csv"
```

执行后会输出：

- 回填后的 `rocks_merged.csv`（默认覆盖输入）
- 未匹配清单 `pyAOBS/utils/rockdata/rock_type_cn_unresolved.csv`

对未匹配项生成半自动中文建议（人工复核后可回填）：

```bash
python pyAOBS/utils/suggest_rock_cn_candidates.py --unresolved "pyAOBS/utils/rockdata/rock_type_cn_unresolved.csv"
```

会生成：

- `pyAOBS/utils/rockdata/rock_type_cn_suggestions.csv`

把你审核后的建议表（csv/xlsx）批量回写到 `rocks_merged.csv`：

```bash
python pyAOBS/utils/apply_rock_cn_suggestions.py --db "pyAOBS/utils/rockdata/rocks_merged.csv" --suggestions "pyAOBS/utils/rockdata/rock_type_cn_suggestions.xlsx"
```
