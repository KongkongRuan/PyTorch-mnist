import torch
import torch.nn.functional as F

# 模型输出
# model_output = torch.tensor([[-2.6791, -0.7951, -0.4575, -1.6953, -2.2006, -0.4585, -1.4573, -0.7502, -3.6072, -4.3196]])

def convert_proba(model_output):
    # 使用Softmax转换为概率
    probabilities = F.softmax(model_output, dim=1)

    # 将概率转换为百分比
    percentages = probabilities * 100

    # 找到概率最大的索引
    max_index = torch.argmax(probabilities, dim=1)

    # 获取概率最高的前三个元素及其索引
    top3_probabilities, top3_indices = torch.topk(probabilities, k=3, dim=1)

    # 打印结果
    for idx, (probs, indices) in enumerate(zip(top3_probabilities[0], top3_indices[0])):
        class_name = f"Number {indices.item()}"
        percentage = f"{probs.item():.4f}%"
        print(f"Sample {idx + 1}: {class_name} with probability {percentage}")