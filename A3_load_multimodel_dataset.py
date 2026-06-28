from datasets import load_from_disk
import matplotlib.pyplot as plt

# 加载数据集（自动识别 DatasetDict 结构）
dataset = load_from_disk("./diy_dataset/my_multimodal_dataset_num15V3")

# 验证结构（推荐）
print("数据集结构:", list(dataset.keys()))  # 应输出 ['train', 'test']
print(f"训练集大小: {len(dataset['train'])} | 测试集大小: {len(dataset['test'])}")
print(f"特征定义: {dataset['train'].features}")

# 访问并可视化第一条训练数据
sample = dataset["train"][0]
print("\n Question:", sample["question"])
print("\n Answer:", sample["answer"])
print("图像类型:", type(sample["image"]))  # 应为 <class 'PIL.Image.Image'>


# 显示图像（自动支持 PIL 对象）
plt.figure(figsize=(6, 6))
plt.imshow(sample["image"])

plt.title(f"Answer: {sample['answer'][:50]}...", fontsize=10)
plt.axis("off")
plt.tight_layout()
plt.show()
