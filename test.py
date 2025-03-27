import random
import matplotlib.pyplot as plt

# 生成1000个x1和x2，范围在0到9之间的整数
data_size = 10000
x1_values = [random.randint(0, 9) for _ in range(data_size)]
x2_values = [random.randint(0, 9) for _ in range(data_size)]

# 计算x1和x2的差值，并取绝对值
absolute_diff_values = [x1 - x2 for x1, x2 in zip(x1_values, x2_values)]

# 统计绝对差值的概率分布
count_dict = {}
for value in absolute_diff_values:
    count_dict[value] = count_dict.get(value, 0) + 1

# 计算概率分布
total_count = sum(count_dict.values())
probability_distribution = {key: count / total_count for key, count in count_dict.items()}

# 绘制概率分布的条形图
plt.bar(probability_distribution.keys(), probability_distribution.values())
plt.xlabel('Absolute Difference')
plt.ylabel('Probability')
plt.title('Probability Distribution of Absolute Differences (x1 - x2)')
plt.show()