import csv
import math


ATTR = 'indo'

# 待处理文件列表(假设当前目录下有这5个文件)
files = [
    f"../prediction/attribution/{ATTR}_brics_cv_0_attribution_summary.csv",
    f"../prediction/attribution/{ATTR}_brics_cv_1_attribution_summary.csv",
    f"../prediction/attribution/{ATTR}_brics_cv_2_attribution_summary.csv",
    f"../prediction/attribution/{ATTR}_brics_cv_3_attribution_summary.csv",
    f"../prediction/attribution/{ATTR}_brics_cv_4_attribution_summary.csv"
]

# 需要求平均的列
avg_cols = ["sub_pred_mean","sub_pred_std","mol_pred_mean","mol_pred_std","attribution","attribution_normalized"]

# 1. 以第一个文件为基准
with open(files[0], "r", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    base_lines = list(reader)

# 构建以smiles为键的映射： {smiles: [行记录列表]}
base_data_by_smiles = {}
for i, row in enumerate(base_lines):
    smi = row["smiles"]
    if smi not in base_data_by_smiles:
        base_data_by_smiles[smi] = []
    base_data_by_smiles[smi].append(row)

# 2. 读取其他文件的数据
all_data = []
for filename in files:
    with open(filename, "r", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        file_data_by_smiles = {}
        for row in reader:
            smi = row["smiles"]
            if smi not in file_data_by_smiles:
                file_data_by_smiles[smi] = []
            file_data_by_smiles[smi].append(row)
        all_data.append(file_data_by_smiles)

# all_data[0] 就是第一个文件的数据（与base_data_by_smiles一致）
# all_data[1:]是其他文件的数据

result_rows = []

# 对每个smiles进行处理
for smi, base_rows_for_smi in base_data_by_smiles.items():
    # 基准文件中该smiles有多少行
    n_lines = len(base_rows_for_smi)
    # 对照其他文件中该smiles的行数是否一致
    for file_idx, file_data_by_sm in enumerate(all_data):
        if smi not in file_data_by_sm:
            raise ValueError(f"在文件 {files[file_idx]} 中缺少 smiles={smi} 的记录")
        if len(file_data_by_sm[smi]) != n_lines:
            raise ValueError(f"行数不匹配: smiles={smi}, 基准文件行数={n_lines}, 文件{files[file_idx]}行数={len(file_data_by_sm[smi])}")

    # 按行进行平均
    for line_idx in range(n_lines):
        # 基准行
        base_row = base_rows_for_smi[line_idx]

        # 从所有文件中取出对应的行
        rows_for_avg = []
        for file_data_by_sm in all_data:
            rows_for_avg.append(file_data_by_sm[smi][line_idx])

        out_row = {
            "smiles": base_row["smiles"],
            "label": base_row["label"],
            "sub_name": base_row["sub_name"],
            "split": base_row["split"]
        }

        # 对avg_cols求平均
        for col in avg_cols:
            values = []
            for r in rows_for_avg:
                val_str = r[col].strip()
                if val_str != "":
                    values.append(float(val_str))
            if len(values) > 0:
                out_row[col] = sum(values)/len(values)
            else:
                out_row[col] = ""

        # 对非平均列从基准文件中取得
        for col in fieldnames:
            if col not in avg_cols and col not in out_row:
                out_row[col] = base_row[col]

        result_rows.append(out_row)

# 按照fieldnames顺序写出
output_file = f"../prediction/attribution/{ATTR}_brics_cv_averaged_attribution_summary.csv"
with open(output_file, "w", newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in result_rows:
        writer.writerow(row)

print("平均结果已保存到:", output_file)
