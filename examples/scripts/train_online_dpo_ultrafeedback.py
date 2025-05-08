# Source: https://huggingface.co/docs/trl/main/en/online_dpo_trainer

from datasets import load_dataset
from trl import OnlineDPOConfig, OnlineDPOTrainer #, PairRMJudge
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification

from trl.trainer.callbacks import LogCompletionsCallback

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# judge = PairRMJudge()
reward_model = AutoModelForSequenceClassification.from_pretrained("trl-lib/Qwen2-0.5B-Reward", num_labels=1)
reward_tokenizer = AutoTokenizer.from_pretrained("trl-lib/Qwen2-0.5B-Reward")

dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")
train_dataset = dataset.select(range(0, 20000))
eval_dataset = dataset.select(range(20000, 21000))

training_args = OnlineDPOConfig(output_dir="Qwen2-0.5B-OnlineDPO",
                                logging_steps=10,
                                per_device_train_batch_size=12,)

trainer = OnlineDPOTrainer(
    model=model,
    # judge=judge,
    reward_model=reward_model,
    reward_processing_class=reward_tokenizer,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

completions_callback = LogCompletionsCallback(trainer, num_prompts=8)
trainer.add_callback(completions_callback)

trainer.train()


# accelerate launch examples/scripts/train_online_dpo.py
