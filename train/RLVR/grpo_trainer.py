import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from datasets import load_dataset
from math_verify import verify, parse

from dataclasses import dataclass
from typing import Optional

from dataset_loader import load_math, load_mmlupro, load_humeval


@dataclass
class MyArguments:
    model_name: str
    output_dir: str
    run_name: str
    learning_rate: float
    beta: float
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
    num_generations: int
    max_prompt_length: int
    max_completion_length: int
    num_train_epochs: int
    save_steps: int
    max_grad_norm: float
    report_to: str
    use_vllm: bool
    vllm_max_model_len: int
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
print(training_args)

def get_dataset():
    train = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    test = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    train = train.map(lambda x: {
        "prompt": x["problem"],
        "answer": last_boxed_only_string(x["solution"]),
        "level": x["level"]
        })

    
    test = test.map(lambda x: {
        "prompt": x["problem"],
        "answer": x["answer"],
        "level": x["level"]
        })
    
    train = train.remove_columns(["problem", "solution", "type"])
    test = test.remove_columns(["problem", "solution", "subject", "unique_id"])
    return train, test

train, test = get_dataset()

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def reward_func(completions, answer, **kwargs):
    def reward(s, gt):
        # add the last boxed tag
        last_boxed = last_boxed_only_string(s)
        if last_boxed is not None:
            s = last_boxed
        try:
            is_correct = verify(parse(s), parse(gt))
            return 1 if is_correct else -1
        except:
            return -1  # parsing/verification failed
    return [reward(c, gt) for c, gt in zip(completions, answer)]


model_path = training_args.model_name if not training_args.resume_from_checkpoint else training_args.checkpoint_path
model_name = AutoModelForCausalLM.from_pretrained(model_path)

grpo_config_args = GRPOConfig(
    output_dir=training_args.output_dir,
    run_name=training_args.run_name,
    learning_rate=training_args.learning_rate,
    resume_from_checkpoint=training_args.resume_from_checkpoint,
    beta=training_args.beta,
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
    num_generations=training_args.num_generations,
    max_prompt_length=training_args.max_prompt_length,
    max_completion_length=training_args.max_completion_length,
    num_train_epochs=training_args.num_train_epochs,
    save_steps=training_args.save_steps,
    max_grad_norm=training_args.max_grad_norm,
    report_to=training_args.report_to,
    use_vllm=training_args.use_vllm,
    vllm_max_model_len=training_args.vllm_max_model_len,
    log_completions=training_args.log_completions,
    max_steps=training_args.max_steps,
    evaluation_strategy="no",
    temperature=0.9,
    top_p=1.0,
    top_k=50
)

trainer = GRPOTrainer(
    model=model_name,
    reward_funcs=[reward_func],
    args=grpo_config_args,
    train_dataset=train,
    eval_dataset=test,
)
trainer.train(resume_from_checkpoint=training_args.checkpoint_path if training_args.resume_from_checkpoint else False)
