# Source: https://huggingface.co/docs/trl/main/en/online_dpo_trainer

"""
Usage:

### python examples/scripts/dpo_online.py \

accelerate launch \
    --num_processes 1 \
    examples/scripts/train_online_dpo_gsm8k.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf  \
    --learning_rate 5.0e-6 \
    --output_dir runs/Llama-2-7b-chat-hf \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --use_peft \
    --max_new_tokens 128 \
    --max_length 256 \
    --logging_steps 10

### UNUSED:
    --warmup_ratio 0.1 \
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl.trainer.callbacks import LogCompletionsCallback
from trl.trainer.judges import BasePairwiseJudge
import re
import torch
from trl import (
    LogCompletionsCallback,
    ModelConfig,
    OnlineDPOConfig,
    OnlineDPOTrainer,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
    get_peft_config,
)

# Source: https://github.com/volcengine/verl/blob/v0.1/verl/utils/reward_score/gsm8k.py
def extract_numerical_answer_gsm8k(solution_str, method='strict') -> float:
    assert method in ['strict', 'flexible']

    if method == 'strict':
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?\\$?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split('#### ')[1].replace(',', '').replace('$', '')
    elif method == 'flexible':
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward if there is no answer
            pass
        else:
            invalid_str = ['', '.']
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    if final_answer is None:
        # print(f"Warning: No numerical answer found in the response text: {solution_str}")
        return None
    return float(final_answer)


def get_gsm8k_reward(response_text: str, ground_truth_answer: float) -> list[float]:
    TOLERANCE = 1e-6

    predicted_answer = extract_numerical_answer_gsm8k(response_text)
    if predicted_answer is None:
        # could not parse a number – count as wrong but keep going
        return(0.0)
    try:
        is_match = abs(float(predicted_answer) - float(ground_truth_answer)) < TOLERANCE
    except Exception as e:            # still malformed
        is_match = False
    return 1.0 if is_match else 0.0


class GSM8KCorrectnessJudge(BasePairwiseJudge):
    def __init__(self, prompt_to_answer):
        super().__init__()
        self.prompt_to_answer = prompt_to_answer

    def judge(self, prompts, completion_pairs):
        # prompts: list of strings
        # completion_pairs: list of string pairs (completion1, completion2); each completion looks like this: [{"role": "assistant", "content": completion}]
        # Return a list of ranks: 0 if completion1 is better, 1 if completion2 is better, None for ties
        results = []

        for prompt, (c1, c2) in zip(prompts, completion_pairs):
            # The prompt includes few-shot demos, so grab the last turn (which is the actual prompt).
            prompt = prompt[-1]["content"]
            c1 = c1[0]["content"]
            c2 = c2[0]["content"]

            # print(f"Prompt: {prompt}")
            # print(f"Completion 1: {c1}")
            # print(f"Completion 2: {c2}")

            ground_truth = self.prompt_to_answer[prompt]
            reward1 = get_gsm8k_reward(c1, ground_truth)
            reward2 = get_gsm8k_reward(c2, ground_truth)

            if reward1 > reward2:
                results.append(0)
            elif reward1 < reward2:
                results.append(1)
            else:
                results.append(None)  # Tie

            # print(f"Reward 1: {reward1}")
            # print(f"Reward 2: {reward2}")
            # print(f"Rank: {results[-1]}")
            # print("-" * 50)

        return results


def run_training():
    parser = TrlParser((ScriptArguments, OnlineDPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "left"

    dataset = load_dataset("gsm8k", "main", split="train")

    # Map to conversational format.
    def to_prompt(example):
        return {
            "prompt": [
                {"role": "system", "content": "You are a math reasoning assistant specialized in solving grade school math word problems. Your task is to provide a clear, step-by-step explanation of the solution and, importantly, to output the final numerical answer on a separate line in exactly the following format:\n“#### <final answer>”\nwhere <final answer> is the numerical result (without any units or extra symbols). Do not include any additional text or formatting on that final line, just a single number."},
                {"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"},
                {"role": "assistant", "content": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72"},
                {"role": "user", "content": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"},
                {"role": "assistant", "content": "Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $10.\n#### 10"},
                # {"role": "user", "content": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?"},
                # {"role": "assistant", "content": "In the beginning, Betty has only 100 / 2 = $50.\nBetty's grandparents gave her 15 * 2 = $30.\nThis means, Betty needs 100 - 50 - 30 - 15 = $5 more.\n#### 5"},
                {"role": "user", "content": example["question"]}
                ]
            }
    dataset = dataset.map(to_prompt, remove_columns=["question"])

    # Map prompts to correct answers. -1 to get the last turn in the few-show conversation.
    prompt_to_answer = { sample["prompt"][-1]["content"]: extract_numerical_answer_gsm8k(sample["answer"]) for sample in dataset }

    train_dataset = dataset.select(range(0, 1000))
    eval_dataset = dataset.select(range(1000, 2000))

    judge = GSM8KCorrectnessJudge(prompt_to_answer)

    trainer = OnlineDPOTrainer(
        model=model,
        judge=judge,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_new_tokens, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)
        trainer.add_callback(completions_callback)

    trainer.train()

    trainer.save_model(training_args.output_dir)


def test_judge() -> None:
    prompt = "What is 2 + 3?"
    prompt_to_answer = {prompt: 5.0}
    judge = GSM8KCorrectnessJudge(prompt_to_answer)

    # 1) first completion is correct
    res = judge.judge(
        [prompt],
        [("#### 5", "#### 4")]
    )
    assert res == [0], f"expected [0], got {res}"

    # 2) second completion is correct
    res = judge.judge(
        [prompt],
        [("#### 8", "#### 5")]
    )
    assert res == [1], f"expected [1], got {res}"

    # 3) tie (both correct)
    res = judge.judge(
        [prompt],
        [("#### 5", "#### 5")]
    )
    assert res == [None], f"expected [None], got {res}"

    print("All GSM8KCorrectnessJudge tests passed.")


if __name__ == "__main__":
    # test_judge()
    run_training()
