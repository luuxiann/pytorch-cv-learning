import os
import pandas as pd
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:  				   
    f.write('NumRooms,Alley,Area,YearBuilt,City,Price\n')  # 列名
    f.write('NA,Pave,1200,1990,Beijing,127500\n')
    f.write('2,NA,800,1985,Shanghai,106000\n')
    f.write('4,NA,1500,2000,Beijing,178100\n')
    f.write('NA,NA,950,1995,NA,140000\n')
    f.write('3,Pave,1800,2010,Guangzhou,210000\n')
    f.write('5,NA,2000,2015,Shanghai,285000\n')
    f.write('NA,Pave,1350,2005,Shenzhen,195000\n')
    f.write('2,NA,1100,1998,NA,125000\n')
    f.write('4,Pave,1650,2012,Beijing,230000\n')
    f.write('3,NA,1400,2008,Guangzhou,175000\n')
data = pd.read_csv(data_file)
print(data)

missing_counts = data.isnull().sum()		# 计算每列的缺失值数量
# 找到缺失值最多的列
max_missing_col = missing_counts.idxmax()
max_missing_count = missing_counts.max()
print(f"缺失值最多的列: '{max_missing_col}'，有 {max_missing_count} 个缺失值")
# 删除缺失值最多的列
data_processed = data.drop(columns=[max_missing_col])
print(f"删除列 '{max_missing_col}' 后的数据:")
print(data_processed)

inputs, outputs = data_processed.iloc[:, :-1], data_processed.iloc[:, -1]
inputs = inputs.fillna(inputs.mean()) 
print(inputs)              
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
X = torch.tensor(inputs.to_numpy(dtype=float))
Y = torch.tensor(outputs.to_numpy(dtype=float))
print(X)
print(Y)