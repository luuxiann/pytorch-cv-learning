import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)  # 固定随机种子
num_samples = 2000  # 2000条租房数据

# 生成特征（现实中影响租金的关键因素）
area = np.clip(np.random.lognormal(3.5, 0.3, num_samples), 30, 150)  # 房屋面积（30-150㎡）
age = np.random.uniform(0, 30, num_samples)  # 房龄（0-30年）
subway_dist = np.clip(np.random.exponential(1.2, num_samples), 0.1, 5)  # 距地铁（0.1-5km）
bedrooms = np.random.choice([1,2,3,4,5], p=[0.1,0.4,0.3,0.15,0.05], size=num_samples)  # 卧室数
is_decorated = np.random.binomial(1, 0.6, num_samples)  # 是否精装（0/1）
district = np.random.choice([0,1,2,3,4], p=[0.1,0.2,0.3,0.25,0.15], size=num_samples)  # 区域（0=核心）

# 生成租金（模拟真实定价逻辑，含非线性关系）
price_area = area * (80 - area * 0.05)  # 面积基础价（面积越大单价略低）
price_age = price_area * (1 - age * 0.005)  # 房龄折价
price_subway = price_age * (1 + (5 - subway_dist) * 0.08)  # 地铁溢价
price_bedroom = price_subway * (1 + bedrooms * 0.15)  # 卧室加价
price_decor = price_bedroom * (1 + is_decorated * 0.2)  # 装修溢价
price_district = price_decor * np.array([1.5,1.3,1.1,0.9,0.7])[district]  # 区域溢价
rent = np.clip(price_district * (1 + np.random.normal(0, 0.1, num_samples)), 1000, 20000)  # 加噪声

# 保存为CSV
data = pd.DataFrame({
    'area': area.round(1),
    'age': age.round(1),
    'subway_dist': subway_dist.round(2),
    'bedrooms': bedrooms,
    'is_decorated': is_decorated,
    'district': district,
    'rent': rent.round(0).astype(int)
})
data.to_csv('house_rent_dataset.csv', index=False)
print(f"生成 'house_rent_dataset.csv'，共{num_samples}条数据")
print("前5行预览：\n", data.head())

# 面积与租金关系可视化
plt.scatter(data['area'], data['rent'], alpha=0.5, s=10)
plt.xlabel('house_areas(m^2)')
plt.ylabel('monthly_rent(yuan)')
plt.title('Area and Rent Relationship')
plt.show()