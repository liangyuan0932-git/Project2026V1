import pandas as pd
from datasets import Dataset, Features, Image, Value
import os, io
from PIL import Image as PilImage  # 用于解码图像
import matplotlib.pyplot as plt

num_videos = 15  # 处理的视频文件数量, < select-code:3-1 >
version = 3     # 数据集版本,  < select-code:3-2 >

# dataset保存路径
save_path = f"./diy_dataset/my_multimodal_dataset_num{num_videos}V{version}"  # 需要修改的输出目录名称
os.makedirs(save_path, exist_ok=True)

# 加载CSV文件
df = pd.read_csv(f"./dataset_num{num_videos}V{version}/num{num_videos}_trajectory.csv")  # 需要修改的CSV文件路径

def gen():
    for _, row in df.iterrows():
        img_path = row["image_path"]  # 假设 CSV 中已包含完整路径
        extrinsic_matrix = row["extrinsic_matrix_str"]
        intrinsic_matrix = row["intrinsic_matrix_str"]
        if os.path.exists(img_path):
            yield {
                # "question": row["question"],  # 固定问题,
                # "Please place the reasoning process between <REASONING> and </REASONING>. 
                "question": "Based on the red navigation route in the image, predict the three-dimensional coordinate trajectory of the key points.\n" \
                "The internal and external parameter matrices of the camera are known.\n" \
                f"Camera extrinsic matrix:\n{extrinsic_matrix}\n" \
                f"Camera intrinsic matrix:\n{intrinsic_matrix}\n" \
                "The final answer lies between <SOLUTION> and </SOLUTION>.\n" \
                "Answers must be formatted in multiple triplets (x, y, z), separated by commas, for example: (0.000, 0.000, 0.000), (3.192, -0.164, -0.019)\n" \
                "Do not output any other content.",
                "image": open(img_path, 'rb').read(),  # 使用open读取二进制内容
                "answer": row["answer"],
            }

features = Features({
    "question": Value("string"),  # 固定问题
    "image": Image(decode=True),  # 设置decode参数为True来自动解码图像
    "answer": Value("string"),
})

dataset = Dataset.from_generator(gen, features=features)  # 生成数据集实例
splits = dataset.train_test_split(test_size=0.4, seed=42) # 划分训练集和测试集，40%作为测试集
print(splits)

# 将数据集保存到磁盘
splits.save_to_disk(save_path)
print(f"数据集已保存至 {save_path}")

# =============================
train_dataset = splits["train"]  # 选择训练集部分
test_dataset = splits["test"]  # 选择测试集部分

# 获取并打印训练集中的第一条数据的描述
print(f"Question: {train_dataset[0]['question']}")
print(f"Answer: {train_dataset[0]['answer']}")

# 解码并显示图像
image_data = train_dataset[0]['image']
if isinstance(image_data, bytes):  # 如果数据是字节流，则转换为图像对象
    image_data = PilImage.open(io.BytesIO(image_data))
plt.imshow(image_data)
plt.axis("off")  # 不显示坐标轴
plt.show()
