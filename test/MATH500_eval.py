import re
import pandas as pd
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams
from math_verify import verify, parse

# ——————————————————————
# Configurations
# ——————————————————————
model_path = ""  # ← Fill this with your model path
csv_path = ""  # ← Fill this with your CSV path if needed
num_trials = 1
temperature = 0.9
top_p = 1.0
top_k = 50

# Prompt templates
PROMPT_TEMPLATES = {
    "system_prompt_1": "{prompt}",
    "system_prompt_2": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ),
    "system_prompt_3": (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively. "
        "User: {prompt}\nAssistant: <think>"
    )
}
PROMPT_TEMPLATE = PROMPT_TEMPLATES["system_prompt_1"]

# ——————————————————————
# Utility functions
# ——————————————————————

def last_boxed_only_string(string: str) -> str | None:
    """Extract the last \boxed{} or \fbox{} expression from a string."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]

    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    brace_count = 0
    for i in range(idx, len(string)):
        if string[i] == "{":
            brace_count += 1
        elif string[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                return string[idx:i + 1]
    return None

def reward_with_format(response: str, ground_truth: str) -> int:
    """Compute reward based on parsed boxed answer."""
    try:
        extracted = last_boxed_only_string(response)
        return int(verify(parse(extracted), parse(ground_truth))) if extracted else 0
    except:
        return 0

def reward_without_format(response: str, ground_truth: str) -> int:
    """Compute reward using full string comparison."""
    try:
        return int(verify(parse(response), parse(ground_truth)))
    except:
        return 0

# ——————————————————————
# Load dataset and build prompts
# ——————————————————————
test_ds = load_dataset("HuggingFaceH4/MATH-500", trust_remote_code=True)["test"]
base_prompts = [PROMPT_TEMPLATE.format(prompt=ex["problem"]) for ex in test_ds]
ground_truths = [last_boxed_only_string(ex["solution"]) for ex in test_ds]

# Duplicate for trials
all_prompts = base_prompts * num_trials
all_ground_truths = ground_truths * num_trials

# ——————————————————————
# Generate responses
# ——————————————————————
llm = LLM(model=model_path, max_model_len=32000, tensor_parallel_size=1)
sampling_params = SamplingParams(
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    max_tokens=32000,
    n=1,
)

print(f"Generating {len(all_prompts)} completions ({num_trials} trial(s) × {len(base_prompts)} prompts)...")
outputs = llm.generate(all_prompts, sampling_params)

# ——————————————————————
# Evaluate and collect rewards
# ——————————————————————
rewards_with_format = []
rewards_without_format = []
responses = []

for out, gt in zip(outputs, all_ground_truths):
    resp = out.outputs[0].text
    responses.append(resp)
    rewards_with_format.append(reward_with_format(resp, gt))
    rewards_without_format.append(reward_without_format(resp, gt))

rewards_with_format = np.array(rewards_with_format).reshape(num_trials, -1)
rewards_without_format = np.array(rewards_without_format).reshape(num_trials, -1)

# ——————————————————————
# Summary statistics
# ——————————————————————
def summarize_scores(name, scores):
    trial_means = scores.mean(axis=1)
    print(f"{name} means per trial:          {trial_means}")
    print(f"Mean of means ({name}):         {np.mean(trial_means):.3f}")
    print(f"Std dev of means ({name}):      {np.std(trial_means):.6f}\n")

print("\n========== FINAL SUMMARY ==========")
summarize_scores("With format", rewards_with_format)
summarize_scores("Without format", rewards_without_format)

# ——————————————————————
# Save results
# ——————————————————————
response_lengths = [len(r) for r in responses]
df = pd.DataFrame({
    "prompt": all_prompts,
    "ground_truth": all_ground_truths,
    "response": responses,
    "response_length": response_lengths
})
# Uncomment to save
df.to_csv(csv_path, index=False)

# ——————————————————————
# Response length statistics
# ——————————————————————
response_lengths_array = np.array(response_lengths).reshape(num_trials, -1)
trial_mean_lengths = response_lengths_array.mean(axis=1)

print("\n========== RESPONSE LENGTH SUMMARY ==========")
print(f"Mean response lengths per trial:       {trial_mean_lengths}")
print(f"Mean of means (response length):       {np.mean(trial_mean_lengths):.3f}")
print(f"Standard deviation of mean lengths:    {np.std(trial_mean_lengths):.6f}")
