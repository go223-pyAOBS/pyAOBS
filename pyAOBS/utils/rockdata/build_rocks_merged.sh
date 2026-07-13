#!/usr/bin/env bash
set -eu
set -o pipefail 2>/dev/null || true

SOURCE_XLSX="${1:-pyAOBS/utils/rockdata/518-1_AppendixAseismicvelocitydata.xlsx}"
BASE_DB="${2:-pyAOBS/utils/rockdata/rocks.xlsx}"
OUT_MERGED="${3:-pyAOBS/utils/rockdata/rocks_merged.csv}"
MIN_PRESSURE_GPA="${MIN_PRESSURE_GPA:-0.1}"
MAX_PRESSURE_GPA="${MAX_PRESSURE_GPA:-0.6}"
SHEET="${SHEET:-Data Master}"
REPORT_TOP="${REPORT_TOP:-30}"

echo "[1/3] 生成 external_rock_data_template.csv ..."
python -m pyAOBS.utils.prepare_external_rock_data --source-csv "$SOURCE_XLSX" --preset p3_518 --sheet "$SHEET" --min-pressure-gpa "$MIN_PRESSURE_GPA" --max-pressure-gpa "$MAX_PRESSURE_GPA" --report-sources --report-top "$REPORT_TOP"

echo
echo "[2/3] 合并生成 rocks_merged.csv ..."
python -m pyAOBS.utils.merge_rock_database --incoming "pyAOBS/utils/rockdata/external_rock_data_template.csv" --base "$BASE_DB" --out "$OUT_MERGED"

echo
echo "[3/3] 健康检查 ..."
python -m pyAOBS.utils.rock_database_health_check --db "$OUT_MERGED" --top 20

echo
echo "完成: $OUT_MERGED"
