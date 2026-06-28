import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "./models"
os.environ["HF_HUB_CACHE"] = "./models"


num_video = 20
version = 3      
dir_base = f"./qwen3_lora_num{num_video}V{version}20"  
FULL_MODEL_DIR = f"{dir_base}/full_tun_model"
image_path = f"dataset_num{num_video}V{version}/frames/0a0fc7a5db365174/frame_000010.jpg" 
txtPath = f"{dir_base}/output.txt"  


# 设置环境变量避免tokenizers并行警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#-----------------------------------------------
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import os
import torch

print("加载微调后的模型...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    FULL_MODEL_DIR,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(FULL_MODEL_DIR, trust_remote_code=True)

# 评估模式
model.eval()
print("Eval mode!!!")

SIMPLIFIED_QUESTION = (
           "Based on the red navigation route in the image, predict the three-dimensional coordinate trajectory of the key points.\n" \
            "The internal and external parameter matrices of the camera are known.\n" \
            "The final answer lies between <SOLUTION> and </SOLUTION>.\n" \
            "Answers must be formatted in multiple triplets (x, y, z), separated by commas, for example: (0.000, 0.000, 0.000), (3.192, -0.164, -0.019)\n" \
            "Do not output any other content.",
        )

messages = [
    {
        "role": "user",
        "content": [
            {
             "type": "image",
             "image": image_path,
            },
            {"type": "text",
             "text": SIMPLIFIED_QUESTION},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=256)  # 1024
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)


with open(txtPath, 'w') as file:
    file.write(str(output_text)) 

print("ALL Done!!!")