import os
os.environ["HF_HOME"] = "./models"
os.environ["HF_HUB_CACHE"] = "./models"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_DATASETS_CACHE"] = "./models/datasets"

import torch, re, math
from unsloth import FastVisionModel
from transformers import AutoProcessor
from datasets import load_from_disk
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

num_video = 20
version = 3     
FULL_MODEL_DIR = f"./qwen3_lora_num{num_video}V{version}/full_tun_model"  
OUTPUT_DIR = f"./unsloth_grpo_outputs_num{num_video}V{version}/logs"      
LORA_PATH = f"./unsloth_grpo_outputs_num{num_video}V{version}/grpo_lora"  
DATASET_PATH = f"./diy_dataset/my_multimodal_dataset_num{num_video}V{version}"    
TEST_OUTPUT_DIR = f"./code_project_num{num_video}V{version}"   
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
TEST_OUTPUT_FILE = f"{TEST_OUTPUT_DIR}/test_results_num{num_video}V{version}.csv" 

BATCH_SIZE = 240 

print("加载测试数据集...")
dataset = load_from_disk(DATASET_PATH)
test_dataset = dataset["test"]

def preprocess_dataset(examples):
    processed = {"images": [], "prompts": [], "answers": [], "indices": []}
    
    for idx, image in enumerate(examples["image"]):
        image = image.resize((256, 256))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": (
                        "Based on the red navigation route in the image, predict the three-dimensional coordinate trajectory of the key points.\n" \
                        "The internal and external parameter matrices of the camera are known.\n" \
                        "The final answer lies between <SOLUTION> and </SOLUTION>.\n" \
                        "Answers must be formatted in multiple triplets (x, y, z), separated by commas, for example: (0.000, 0.000, 0.000), (3.192, -0.164, -0.019)\n" \
                        "Do not output any other content.",
                    )}
                ]
            }
        ]
        prompt_str = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        processed["images"].append(image)
        processed["prompts"].append(prompt_str)
        processed["answers"].append(examples["answer"][idx])
        processed["indices"].append(idx)
    
    return processed

print("加载基础模型和processor...")
processor = AutoProcessor.from_pretrained(FULL_MODEL_DIR)
processor.tokenizer.pad_token = processor.tokenizer.eos_token

print("预处理测试集...")
test_dataset = test_dataset.map(
    preprocess_dataset,
    batched=True,
    batch_size=1000,  
    remove_columns=test_dataset.column_names
)

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=FULL_MODEL_DIR,
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=torch.float16,
    gpu_memory_utilization=0.85,
)

print("加载LoRA适配器...")
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    use_gradient_checkpointing="unsloth",
)

model.load_adapter(LORA_PATH, adapter_name="default")
model.eval()  

SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

def parse_coordinates(text):
    """从字符串中提取所有(x, y, z)三元组，返回浮点数列表"""
    pattern = r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)'
    matches = re.findall(pattern, text)
    return [(float(x), float(y), float(z)) for x, y, z in matches]

def compute_average_distance(pred_points, true_points):
    """计算预测坐标点与真实坐标点的平均欧氏距离"""
    if isinstance(true_points, str):
        true_points = parse_coordinates(true_points)
    
    if not pred_points or not true_points:
        return float('nan')
    
    n = min(len(pred_points), len(true_points))
    total_dist = 0.0
    for p, t in zip(pred_points[:n], true_points[:n]):
        total_dist += math.sqrt(sum((a-b)**2 for a,b in zip(p, t)))
    
    return total_dist / n if n > 0 else float('nan')

def extract_answer_from_completion(completion):
    """从模型生成的文本中提取<SOLUTION>标签内的内容"""
    pattern = f'{SOLUTION_START}(.*?){SOLUTION_END}'
    matches = re.findall(pattern, completion, re.DOTALL)
    return matches[0].strip() if matches else completion

print(f"\n开始批量测试 (batch_size={BATCH_SIZE})...")
results = []

dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=lambda x: {
        "images": [sample["images"] for sample in x],
        "prompts": [sample["prompts"] for sample in x],
        "answers": [sample["answers"] for sample in x],
        "indices": [sample["indices"] for sample in x],
    }
)

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Processing batches"):
        inputs = processor(
            images=batch["images"],
            text=batch["prompts"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False
        ).to("cuda")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            do_sample=False,  
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        
        input_length = inputs["input_ids"].shape[1]
        generated_ids = outputs[:, input_length:]
        
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        for i in range(len(batch["indices"])):
            idx = batch["indices"][i]
            true_answer = batch["answers"][i]
            generated_text = generated_texts[i]
            
            final_answer = extract_answer_from_completion(generated_text)
            pred_points = parse_coordinates(final_answer)
            
            avg_dist = compute_average_distance(pred_points, true_answer)
            
            results.append({
                "index": str(idx),
                "true_answer": true_answer,
                "generated_text": generated_text,
                "extracted_answer": final_answer,
                "avg_distance": avg_dist
            })
        
        torch.cuda.empty_cache()

df = pd.DataFrame(results)
df.to_csv(TEST_OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"\n测试完成, 结果已保存至 {TEST_OUTPUT_FILE}")
print(f"平均欧氏距离: {df['avg_distance'].mean():.4f} ± {df['avg_distance'].std():.4f}")