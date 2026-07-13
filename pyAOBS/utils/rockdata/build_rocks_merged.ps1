param(
    [string]$SourceXlsx = "pyAOBS/utils/rockdata/518-1_AppendixAseismicvelocitydata.xlsx",
    [string]$BaseDb = "pyAOBS/utils/rockdata/rocks.xlsx",
    [string]$OutMerged = "pyAOBS/utils/rockdata/rocks_merged.csv",
    [double]$MinPressureGpa = 0.1,
    [double]$MaxPressureGpa = 0.6,
    [string]$Sheet = "Data Master",
    [int]$ReportTop = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Write-Host "[1/3] 生成 external_rock_data_template.csv ..."
python -m pyAOBS.utils.prepare_external_rock_data `
  --source-csv $SourceXlsx `
  --preset p3_518 `
  --sheet $Sheet `
  --min-pressure-gpa $MinPressureGpa `
  --max-pressure-gpa $MaxPressureGpa `
  --report-sources `
  --report-top $ReportTop

Write-Host ""
Write-Host "[2/3] 合并生成 rocks_merged.csv ..."
python -m pyAOBS.utils.merge_rock_database `
  --incoming "pyAOBS/utils/rockdata/external_rock_data_template.csv" `
  --base $BaseDb `
  --out $OutMerged

Write-Host ""
Write-Host "[3/3] 健康检查 ..."
python -m pyAOBS.utils.rock_database_health_check `
  --db $OutMerged `
  --top 20

Write-Host ""
Write-Host "完成: $OutMerged"
