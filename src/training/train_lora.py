from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from src.utils.data_loader import load_and_tokenize

# 데이터셋 로드
dataset = load_dataset("json", data_files="data/dev_docs.json")["train"]

# 토크나이저 & 데이터 전처리
tokenized_dataset, tokenizer = load_and_tokenize(dataset)

# 모델 로드 (CPU)
model = AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-j", device_map="cpu")

# LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# 학습 설정
training_args = TrainingArguments(
    output_dir="models/lora",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    logging_steps=5,
    save_steps=20,
    save_total_limit=2,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()
