import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings
warnings.filterwarnings("ignore")

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,                 # <--- 我们换回了最经典的原生 Trainer
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, TaskType, get_peft_model

# --- 第一部分：基础配置 ---
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_FILE = "shark_identity.json"
OUTPUT_DIR = "./shark_lora_output"

print(f"🚀 准备开始训练 (原生稳健版)！目标模型：{MODEL_NAME}")

# --- 第二部分：加载模型 ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

print("正在加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# 补全 padding token，防止报错
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0}, 
    trust_remote_code=True
)

# --- 第三部分：配置 LoRA ---
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
# 显式给模型挂载 LoRA，不依赖 Trainer 自动挂载
model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # 打印一下看看有多少参数要练

# --- 第四部分：数据处理 (最稳的手动 Tokenize) ---
print("正在处理数据...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

def process_data(example):
    # 1. 拼文本
    instruction = example['instruction']
    output = example['output']
    text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
    
    # 2. 变成数字 (Tokenize)
    # max_length设为 512，防止显存爆
    tokenized = tokenizer(text, truncation=True, max_length=512)
    
    # 3. 构造 labels (对于自回归模型，labels 就是 input_ids)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# 手动处理所有数据
tokenized_dataset = dataset.map(process_data, remove_columns=dataset.column_names)

# --- 第五部分：训练参数 ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=5,
    max_steps=60,                  
    save_steps=60,
    fp16=True,
    optim="paged_adamw_8bit",      
    report_to="none"               
)

# --- 第六部分：开始训练 ---
# 使用通用的 DataCollator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    data_collator=data_collator
)

print("\n🔥🔥🔥 开始炼丹！请盯着你的显存看！🔥🔥🔥")
trainer.train()

# 保存模型
trainer.model.save_pretrained(OUTPUT_DIR)
print(f"\n✅ 训练完成！LoRA 权重已保存在 {OUTPUT_DIR}")