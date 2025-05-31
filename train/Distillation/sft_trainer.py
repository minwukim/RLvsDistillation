import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from datasets import load_dataset
from math_verify import verify, parse
from custom_MATH_reward import compute_score, remove_boxed, last_boxed_only_string

from dataclasses import dataclass
from typing import Optional

from dataset_loader import load_math, load_countdown, load_kk
import pandas as pd

@dataclass
class MyArguments:
    model_name: str
    output_dir: str
    run_name: str
    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    weight_decay: float
    warmup_steps: int
    lr_scheduler_type: str
    logging_steps: float
    bf16: bool
    bf16_full_eval: bool
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    max_prompt_length: int
    max_completion_length: int
    num_train_epochs: int
    save_steps: int
    max_grad_norm: float
    report_to: str
    log_completions: bool
    checkpoint_path: str = None
    resume_from_checkpoint: bool = False
    max_steps: int = -1
    eval_on_start: bool = False
    eval_steps: int = None
    evaluation_strategy: str = None


from trl import TrlParser

parser = TrlParser(dataclass_types=[MyArguments])

training_args = parser.parse_args_and_config()[0]

train_set = None
test_set = None


# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_set.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_set.reset_index(drop=True))


model_path = training_args.model_name if not training_args.resume_from_checkpoint else training_args.checkpoint_path
model_name = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side='left'

if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

collator = DataCollatorForCompletionOnlyLM(
    response_template=tokenizer.encode(" Assistant: <think>\n", add_special_tokens=False),
    tokenizer=tokenizer,
)

sft_config_args = SFTConfig(
    output_dir=training_args.output_dir,
    run_name=training_args.run_name,
    learning_rate=training_args.learning_rate,
    resume_from_checkpoint=training_args.resume_from_checkpoint,
    adam_beta1=training_args.adam_beta1,
    adam_beta2=training_args.adam_beta2,
    weight_decay=training_args.weight_decay,
    warmup_steps=training_args.warmup_steps,
    lr_scheduler_type=training_args.lr_scheduler_type,
    logging_steps=training_args.logging_steps,
    bf16=training_args.bf16,
    bf16_full_eval=training_args.bf16_full_eval,
    per_device_train_batch_size=training_args.per_device_train_batch_size,
    gradient_accumulation_steps=training_args.gradient_accumulation_steps,
    gradient_checkpointing=training_args.gradient_checkpointing,
    max_seq_length=training_args.max_completion_length,
    num_train_epochs=training_args.num_train_epochs,
    save_steps=training_args.save_steps,
    max_grad_norm=training_args.max_grad_norm,
    report_to=training_args.report_to,
    max_steps=training_args.max_steps,
    eval_strategy=training_args.evaluation_strategy,
    eval_steps = training_args.eval_steps,
    eval_on_start=training_args.eval_on_start,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    per_device_eval_batch_size=training_args.per_device_train_batch_size,
    eval_accumulation_steps=training_args.gradient_accumulation_steps
)

trainer = SFTTrainer(
    model=model_name,
    args=sft_config_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=collator
)

trainer.train(resume_from_checkpoint=training_args.checkpoint_path if training_args.resume_from_checkpoint else False)

