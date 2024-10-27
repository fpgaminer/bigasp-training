#!/usr/bin/env python3
from transformers.trainer_callback import TrainerControl, TrainerState
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import json
from unsloth.chat_templates import get_chat_template, train_on_responses_only, _find_common_token_ids
from pathlib import Path
import datasets
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import wandb
from transformers import TrainerCallback
from functools import wraps
from typing import Union, Any, Optional, List
import torch.nn as nn
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import nested_detach
import types
from transformers.integrations import WandbCallback, rewrite_logs
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--device-batch-size', type=int, default=16, help='Batch size per device')
parser.add_argument('--desired-batch-size', type=int, default=64, help='Desired batch size')
parser.add_argument('--warmup-samples', type=int, default=160, help='Warmup samples')
parser.add_argument('--train-samples', type=int, default=3200, help='Train samples')
parser.add_argument('--eval-samples', type=int, default=1600, help='Eval samples')
parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
parser.add_argument('--train-path', type=Path, default=Path('train.jsonl'), help='Path to the training data')
parser.add_argument('--test-path', type=Path, default=Path('test.jsonl'), help='Path to the test data')
parser.add_argument('--inject-system-message', action='store_true', help='Inject the system message into the chat template')
parser.add_argument('--lora-rank', type=int, default=64, help='Rank of the LoRA matrix')



def main():
	args = parser.parse_args()


	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
		max_seq_length=2048,
		dtype=torch.bfloat16,
		load_in_4bit=False,
	)

	model = FastLanguageModel.get_peft_model(
		model,
		r=args.lora_rank,
		target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
		lora_alpha=16,
		lora_dropout=0, # Supports any, but = 0 is optimized
		bias="none",    # Supports any, but = "none" is optimized
		use_gradient_checkpointing="unsloth",
		random_state=3407,
		use_rslora = False,  # We support rank stabilized LoRA
		loftq_config = None, # And LoftQ
	)

	tokenizer = get_chat_template(
		tokenizer,
		chat_template = "llama-3.1",
	)

	ds = load_dataset("json", data_files={'train': str(args.train_path), 'test': str(args.test_path)})
	assert isinstance(ds, datasets.DatasetDict)

	print(ds['train'][5]['messages'])

	def formatting_prompts_func(examples):
		convos = examples["messages"]
		texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
		return { "text": texts }

	ds = ds.map(formatting_prompts_func, batched = True,)

	print(ds['train'][5]['text'])

	wandb.init(project="sdxl-big-asp-recaption")

	# Add config
	wandb.config.update(args, allow_val_change=True)

	#warmup_samples = 160
	#train_samples = 3200
	#eval_samples = 1600
	#desired_batch_size = 64
	#device_batch_size = 16

	accumulation_steps = args.desired_batch_size // args.device_batch_size
	warmup_steps = args.warmup_samples // args.desired_batch_size
	train_steps = args.train_samples // args.desired_batch_size
	eval_steps = args.eval_samples // args.desired_batch_size

	#callback = OurCallback()
	callback = OurWandbCallback(samples_per_step = args.desired_batch_size)

	# TODO: NEFT
	trainer = SFTTrainer(
		model = model,
		tokenizer = tokenizer,
		train_dataset = ds['train'],
		eval_dataset = ds['test'],
		dataset_text_field = "text",
		data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
		dataset_num_proc = 2,
		packing = False,
		callbacks = [callback],
		args = TrainingArguments(
			per_device_eval_batch_size = args.device_batch_size,
			per_device_train_batch_size = args.device_batch_size,
			eval_accumulation_steps = accumulation_steps,
			gradient_accumulation_steps = accumulation_steps,
			warmup_steps = warmup_steps,
			max_steps = train_steps,
			learning_rate = args.learning_rate,
			bf16 = True,
			logging_steps = 1,
			optim = "adamw_torch",
			weight_decay = 0.01,
			lr_scheduler_type = "cosine",
			seed = 3407,
			output_dir = "outputs",
			evaluation_strategy="steps",
			eval_steps = eval_steps,
			do_eval=True,
			#neftune_noise_alpha=5,
		),
	)

	trainer = our_train_on_responses_only(
		trainer,
		instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
		response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
	)

	#for callback in trainer.callback_handler.callbacks:
	#	if isinstance(callback, WandbCallback):
	#		print("INFO!!!!! Found WandbCallback, patching")
	#		callback.on_log = types.MethodType(our_wandb_on_log, callback)
	#		#reporting_to_wandb = True
	#		break

	print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))

	space = tokenizer(" ", add_special_tokens = False).input_ids[0]
	print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]))

	trainer_stats = trainer.train()
	print(trainer_stats)

	# Inject the system message into the chat template before saving
	if args.inject_system_message:
		assert ds['train'][0]['messages'][0]['role'] == "system"
		system_message = ds['train'][0]['messages'][0]['content']
		tokenizer.chat_template = tokenizer.chat_template.replace("{{- system_message }}", '{{- ' + json.dumps(system_message) + ' }}')

	model.save_pretrained(f"lora_model_{wandb.run.id}")
	tokenizer.save_pretrained(f"lora_model_{wandb.run.id}")

	# Copy the test data, for reference
	(Path(f"lora_model_{wandb.run.id}") / "test.jsonl").write_text(Path("test.jsonl").read_text())


def our_train_on_responses_only(
    trainer,
    instruction_part = None,
    response_part    = None,
):
	"""
	Trains only on responses and not on the instruction by masking out
	the labels with -100 for the instruction part.
	"""
	tokenizer = trainer.tokenizer

	if  not hasattr(tokenizer, "_unsloth_input_part") or \
		not hasattr(tokenizer, "_unsloth_output_part"):
		
		if instruction_part is None or response_part is None:
			raise ValueError("Unsloth: instruction_part and response_part must be given!")
		pass
	elif (instruction_part is not None or response_part is not None) and \
		(hasattr(tokenizer, "_unsloth_input_part") or hasattr(tokenizer, "_unsloth_output_part")):

		raise ValueError("Unsloth: Your tokenizer already has instruction and response parts set - do not give custom ones!")
	else:
		instruction_part = tokenizer._unsloth_input_part
		response_part    = tokenizer._unsloth_output_part
	pass

	# Get most common tokens since tokenizers can tokenize stuff differently!
	Q_must, Q_left, Q_right = _find_common_token_ids(instruction_part, tokenizer)
	A_must, A_left, A_right = _find_common_token_ids(response_part,    tokenizer)

	# Store some temporary stuff
	A_first = A_must[0]
	len_A_must = len(A_must)
	A_left_reversed = A_left[::-1]
	A_right_forward = A_right

	Q_first = Q_must[0]
	len_Q_must = len(Q_must)
	Q_left_reversed = Q_left[::-1]
	Q_right_forward = Q_right

	def _train_on_responses_only(examples):
		input_ids_ = examples["input_ids"]
		all_labels = []

		for input_ids in input_ids_:
			n = len(input_ids)
			labels = [-100] * n
			n_minus_1 = n - 1
			j = 0
			while j < n:
				# Find <assistant>
				if (input_ids[j] == A_first) and \
					(input_ids[j : (k := j + len_A_must)] == A_must):

					# Now backtrack to get previous optional tokens
					for optional_left in A_left_reversed:
						if j < 1: break
						if optional_left == input_ids[j-1]: j -= 1
						else: break
					pass
					# And forwards look as well
					for optional_right in A_right_forward:
						if k >= n_minus_1: break
						if optional_right == input_ids[k+1]: k += 1
						else: break
					pass
					# assistant_j = j
					assistant_k = k

					j = assistant_k
					# Given <assistant>, now find next user
					while j < n:
						# Find <user>
						# Also accept last final item if assistant is the last turn
						if (j == n_minus_1) or \
							((input_ids[j] == Q_first) and \
								(input_ids[j : (k := j + len_Q_must)] == Q_must)):

							# Now backtrack to get previous optional tokens
							for optional_left in Q_left_reversed:
								if j < 1: break
								if optional_left == input_ids[j-1]: j -= 1
								else: break
							pass
							# And forwards look as well
							for optional_right in Q_right_forward:
								if k >= n_minus_1: break
								if optional_right == input_ids[k+1]: k += 1
								else: break
							pass
							user_j = j
							# Account for last item
							if user_j != n_minus_1:
								# user_k = k
								# j = user_k
								j = k
							else:
								user_j = n
								k = n
							pass
							# Now copy input_ids to labels
							labels[assistant_k : user_j] = input_ids[assistant_k : user_j]
							# print(assistant_j, assistant_k, user_j, user_k)
							break
						pass
						j += 1
					pass
				pass
				j += 1
			pass
			all_labels.append(labels)
		pass
		return { "labels" : all_labels }
	pass
	trainer.train_dataset = trainer.train_dataset.map(_train_on_responses_only, batched = True)
	trainer.eval_dataset = trainer.eval_dataset.map(_train_on_responses_only, batched = True)
	return trainer


class OurWandbCallback(WandbCallback):
	def __init__(self, samples_per_step: int):
		super().__init__()
		self.samples_per_step = samples_per_step
	
	def on_log(self, args, state, control, model=None, logs=None, **kwargs):
		single_value_scalars = [
			"train_runtime",
			"train_samples_per_second",
			"train_steps_per_second",
			"train_loss",
			"total_flos",
		]

		#print(f"INFO: Logging to wandb, state: {state}")

		if self._wandb is None:
			return
		if not self._initialized:
			self.setup(args, state, model)
		if state.is_world_process_zero:
			for k, v in logs.items():
				if k in single_value_scalars:
					self._wandb.run.summary[k] = v
			non_scalar_logs = {k: v for k, v in logs.items() if k not in single_value_scalars}
			non_scalar_logs = rewrite_logs(non_scalar_logs)
			self._wandb.log({**non_scalar_logs, "train/global_step": state.global_step, "train/samples": state.global_step * self.samples_per_step})


class OurCallback(TrainerCallback):
	def __init__(self):
		self.total_optimizer_steps = 0
	
	def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
		self.total_optimizer_steps += 1
		print(f"INFO: Optimizer step, step: {self.total_optimizer_steps}")


if __name__ == '__main__':
	main()