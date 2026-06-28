'''<Intruction> 
train --> merge --> eval --> compared_plot -->logPlot'''

import os
os.environ["HF_HOME"] = "./models"
os.environ["HF_HUB_CACHE"] = "./models"

#-----------------------------------------------
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import json
from PIL import Image
from torch.utils.data import Dataset, Subset
import random
from transformers import TrainingArguments, Trainer, TrainerCallback
from peft import LoraConfig, get_peft_model, PeftModel
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


num_video = 15   
version = 3      
DATASET_PATH = f"./dataset_num{num_video}V{version}/json_dataset.json"  
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"

dir_base = f"./qwen3_lora_num{num_video}V{version}20"  
LOG_DIR = os.path.join(dir_base, "logs")
BEST_MODEL_DIR = os.path.join(dir_base, "best_model")  
OFFLOAD_DIR = "./offload"  

os.makedirs(dir_base, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")

def split_dataset(dataset, train_ratio=0.85, val_ratio=0.15, seed=42):
    assert train_ratio + val_ratio <= 1.0
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_end = int(train_ratio * len(dataset))
    val_end = train_end + int(val_ratio * len(dataset))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    return train_subset, val_subset

class TrajectoryCaptionDataset(Dataset):
    def __init__(self, json_path, processor, max_length=1024):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length
        self.task_field = "Trajectory"
        self.image_token_id = 151652

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        extrinsic_matrix = item["extrinsic_matrix"]
        intrinsic_matrix = item["intrinsic_matrix"]
        
        caption_text = item["caption"].get(self.task_field, "")
        if not caption_text:
            caption_text = "No trajectory data available"
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"警告: 加载图像 {image_path} 时出错: {e}. 使用空白图像替代.")
            image = Image.new("RGB", (256, 256))  # Fallback to a blank image

        '''SIMPLIFIED_QUESTION = (
            "Based on the red navigation route in the image, predict the three-dimensional coordinate trajectory of the key points.\n" \
            "The internal and external parameter matrices of the camera are known.\n" \
            f"Camera extrinsic matrix:\n{extrinsic_matrix}\n" \
            f"Camera intrinsic matrix:\n{intrinsic_matrix}\n" \
            "The final answer lies between <SOLUTION> and </SOLUTION>.\n" \
            "Answers must be formatted in multiple triplets (x, y, z), separated by commas, for example: (0.000, 0.000, 0.000), (3.192, -0.164, -0.019)\n" \
            "Do not output any other content.",
        )'''

        SIMPLIFIED_QUESTION = (
            "Based on the red navigation route in the image, predict the three-dimensional coordinate trajectory of the key points.\n" \
            "The final answer lies between <SOLUTION> and </SOLUTION>.\n" \
            "Answers must be formatted in multiple triplets (x, y, z), separated by commas, for example: (0.000, 0.000, 0.000), (3.192, -0.164, -0.019)\n" \
            "Do not output any other content.",
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": SIMPLIFIED_QUESTION},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": caption_text}],
            },
        ]
        
        text = self.processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False,
        )
        
        encoding = self.processor(
            images=image,
            text=text,
            return_tensors="pt",
            padding="max_length",  
            truncation=True,      
            max_length=self.max_length,
        )

        if "image_grid_thw" not in encoding:
            raise ValueError("关键错误: Processor 未返回 'image_grid_thw'. "
                             "这表明处理器或模型版本不兼容. "
                             "请确保使用来自 Hugging Face 的正确模型和处理器.")

        # Squeeze to remove batch dimension added by processor
        for k in encoding:
            encoding[k] = encoding[k].squeeze(0) 

        # Labels for language modeling (next token prediction)
        encoding["labels"] = encoding["input_ids"].clone()
        return encoding  

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)

model.gradient_checkpointing_enable()  # Enable gradient checkpointing to save memory
model.enable_input_require_grads()     # Crucial for gradient checkpointing with PEFT
model.config.use_cache = False         

# LoRA Configuration targeting relevant modules for VL tasks
lora_config = LoraConfig(
    r=16,             
    lora_alpha=32,    
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",      # Attention layers
        "gate_proj", "up_proj", "down_proj",         # MLP layers
        "visual_projection",                         # Specific to Qwen-VL series visual-language bridge
    ],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  


full_dataset = TrajectoryCaptionDataset(DATASET_PATH, processor, max_length=4096)

train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.85, val_ratio=0.15, seed=42)

try:
    test_sample = train_dataset[0]
    print(f"样本键: {list(test_sample.keys())}")
    print(f"input_ids 形状: {test_sample['input_ids'].shape}")
    print(f"pixel_values 形状: {test_sample['pixel_values'].shape}")
    print(f"image_grid_thw 形状: {test_sample['image_grid_thw'].shape}")
    
    # 使用硬编码的图像token ID (151652)
    image_token_id = 151652
    num_image_tokens = (test_sample['input_ids'] == image_token_id).sum().item()
    print(f"输入中潜在的图像token数量: {num_image_tokens}")
except Exception as e:
    print(f"测试数据集样本时出错: {e}")
    raise

class BestModelCallback(TrainerCallback):
    def __init__(self, best_model_dir, processor):
        self.best_eval_loss = float('inf')  
        self.best_model_dir = best_model_dir
        self.processor = processor
        self.best_epoch = -1
        self.best_step = -1
        self.eval_history = [] 
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and 'eval_loss' in metrics:
            eval_loss = metrics['eval_loss']
            current_epoch = state.epoch
            current_step = state.global_step
            
            self.eval_history.append({
                "epoch": current_epoch,
                "step": current_step,
                "eval_loss": eval_loss,
                "is_best": eval_loss < self.best_eval_loss
            })
            
            history_path = os.path.join(self.best_model_dir, "eval_history.json")
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.eval_history, f, indent=2, ensure_ascii=False)
            
            if eval_loss < self.best_eval_loss:
                best_eval_loss_before = self.best_eval_loss  
                self.best_eval_loss = eval_loss
                self.best_epoch = current_epoch
                self.best_step = current_step
                
                print(f"\n{'='*50}")
                print(f"发现新最佳模型! 验证损失: {eval_loss:.4f} (Previous: {best_eval_loss_before:.4f})")
                print(f"Epoch: {current_epoch:.2f} | Step: {current_step}")
                print(f"保存到: {self.best_model_dir}")
                print(f"{'='*50}\n")
                
                model = kwargs['model']
                model.save_pretrained(self.best_model_dir)
                self.processor.save_pretrained(self.best_model_dir)
                
                config = model.config.to_dict()
                config_path = os.path.join(self.best_model_dir, 'config.json')
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, ensure_ascii=False, indent=4)


training_args = TrainingArguments(
    output_dir=dir_base,
    logging_dir=LOG_DIR,
    per_device_train_batch_size=1,       
    gradient_accumulation_steps=4,       
    per_device_eval_batch_size=2,        
    learning_rate=5e-4,                   
    num_train_epochs=8,                 
    logging_steps=10,                    
    eval_accumulation_steps=2,           

    eval_strategy="steps",               
    save_strategy="steps",              
    eval_steps = 20,                    
    save_steps = 20,                                   
    remove_unused_columns=False,         
    fp16=not torch.cuda.is_bf16_supported(), 
    bf16=torch.cuda.is_bf16_supported(),     
    dataloader_num_workers=8,            
    report_to="tensorboard",             
    optim="adamw_torch",
    lr_scheduler_type="cosine",         
    warmup_ratio=0.1,                    
    weight_decay=0.01,                   
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, 
    load_best_model_at_end=True,         
    metric_for_best_model="eval_loss",  
    greater_is_better=False,             
    run_name="qwen3_vl_lora_finetune",   
    save_total_limit=3,                 
)


best_model_callback = BestModelCallback(BEST_MODEL_DIR, processor)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    callbacks=[best_model_callback]  
)


print("\n" + "="*60)
print("开始训练...")
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"最佳模型将保存到: {BEST_MODEL_DIR}")
print("训练过程中将每个epoch后评估验证损失, 仅当验证损失改善时保存模型")
print("="*60 + "\n")

train_result = trainer.train() # (resume_from_checkpoit=True)

print(f"\n{'='*60}")
print("训练完成!")
print(f"历史最佳验证损失: {best_model_callback.best_eval_loss:.4f}")
print(f"最佳模型出现在 Epoch: {best_model_callback.best_epoch:.2f} | Step: {best_model_callback.best_step}")
print(f"最佳模型已保存至: {BEST_MODEL_DIR}")
print(f"训练日志保存至: {LOG_DIR}")
print(f"{'='*60}\n")

metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)
metrics["eval_samples"] = len(val_dataset)

trainer.save_metrics("all", metrics)
trainer.save_state()
