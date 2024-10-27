#!/usr/bin/env python3
"""
Use JoyCaption to caption all the images.
Usage: torchrun --standalone --nproc_per_node=8 caption-images.py
"""
import torch
from tqdm import tqdm
from PIL import Image
import PIL.Image
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import argparse
import torchvision.transforms.functional as TVF
from pathlib import Path
import torch.amp
import psycopg
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, LlavaForConditionalGeneration
import random
import numpy as np
from typing import Iterator, Optional
import torch.distributed
import logging
from torch.distributed.elastic.multiprocessing.errors import record


PIL.Image.MAX_IMAGE_PIXELS = 933120000
JOYCAPTION_PATH = Path("fancyfeast/llama-joycaption-alpha-two-hf-llava")
MEAN_WORDS = 60 * 3 // 4
STD_WORDS = 40 * 3 // 4


CAPTION_TYPE_MAP = {
	"Descriptive": [
		"Write a descriptive caption for this image in a formal tone.",
		"Write a descriptive caption for this image in a formal tone within {word_count} words.",
		"Write a {length} descriptive caption for this image in a formal tone.",
	],
	"Descriptive (Informal)": [
		"Write a descriptive caption for this image in a casual tone.",
		"Write a descriptive caption for this image in a casual tone within {word_count} words.",
		"Write a {length} descriptive caption for this image in a casual tone.",
	],
	"Training Prompt": [
		"Write a stable diffusion prompt for this image.",
		"Write a stable diffusion prompt for this image within {word_count} words.",
		"Write a {length} stable diffusion prompt for this image.",
	],
	"MidJourney": [
		"Write a MidJourney prompt for this image.",
		"Write a MidJourney prompt for this image within {word_count} words.",
		"Write a {length} MidJourney prompt for this image.",
	],
	"Booru tag list": [
		"Write a list of Booru tags for this image.",
		"Write a list of Booru tags for this image within {word_count} words.",
		"Write a {length} list of Booru tags for this image.",
	],
	"Booru-like tag list": [
		"Write a list of Booru-like tags for this image.",
		"Write a list of Booru-like tags for this image within {word_count} words.",
		"Write a {length} list of Booru-like tags for this image.",
	],
	"Art Critic": [
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it within {word_count} words.",
		"Analyze this image like an art critic would with information about its composition, style, symbolism, the use of color, light, any artistic movement it might belong to, etc. Keep it {length}.",
	],
	"Product Listing": [
		"Write a caption for this image as though it were a product listing.",
		"Write a caption for this image as though it were a product listing. Keep it under {word_count} words.",
		"Write a {length} caption for this image as though it were a product listing.",
	],
	"Social Media Post": [
		"Write a caption for this image as if it were being used for a social media post.",
		"Write a caption for this image as if it were being used for a social media post. Limit the caption to {word_count} words.",
		"Write a {length} caption for this image as if it were being used for a social media post.",
	],
}


parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, default=16)
parser.add_argument("--batch-size", type=int, default=32)


@torch.no_grad()
@record
def main():
	device = f"cuda:{torch.cuda.current_device()}"

	# Logging
	logger = logging.getLogger(f'Process-{distributed_rank()}')
	logging.basicConfig(format='%(asctime)s [%(name)s] [%(levelname)s] [%(funcName)s] - %(message)s')
	logger.setLevel(logging.INFO)

	# Parse arguments
	args = parser.parse_args()

	# Make sure unique seeds are used for each process
	seed = hash((42, distributed_rank())) % 2**32
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)

	# Connect to the database
	conn = psycopg.connect(dbname='postgres', user='postgres', host=str(Path.cwd() / "pg-socket"))
	cur = conn.cursor()

	if distributed_rank() == 0:
		# Fetch a list of all paths we need to work on
		cur.execute("SELECT path FROM images WHERE embedding IS NOT NULL AND caption IS NULL AND score IS NOT NULL AND score > 0")
		paths = [path for path, in cur.fetchall()]
		random.shuffle(paths)  # Absolutely necessary to break associations between database order and prompt (since we use the same prompt for a whole batch)

		# Broadcast the paths to all other processes
		torch.distributed.broadcast_object_list([paths])
	else:
		objects = [None]
		logger.info(f"Rank {distributed_rank()} waiting for paths...")
		torch.distributed.broadcast_object_list(objects)
		paths, = objects

		if paths is None:
			logger.info(f"Rank {distributed_rank()} exiting...")
			return

	# Useful information and sign of life from each rank
	logger.info(f"Rank {distributed_rank()} working on {len(paths)} images...")

	# Load JoyCaption
	tokenizer = AutoTokenizer.from_pretrained(JOYCAPTION_PATH, use_fast=True)
	llava_model = LlavaForConditionalGeneration.from_pretrained(JOYCAPTION_PATH, torch_dtype="bfloat16", device_map=distributed_rank())
	assert isinstance(llava_model, LlavaForConditionalGeneration)

	dataset = ImageDataset(paths, tokenizer, llava_model.config.image_token_index, llava_model.config.image_seq_length)
	sampler = BatchSamplerWithPrompts(dataset, num_replicas=distributed_world_size(), rank=distributed_rank(), batch_size=args.batch_size)
	dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=args.num_workers)

	pbar = tqdm(total=len(paths) // distributed_world_size(), desc="Captioning images...", dynamic_ncols=True, position=distributed_rank())
	for batch in dataloader:
		#print(batch['pixel_values'].shape, batch['convo_tokens'].shape)

		vision_dtype = llava_model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype

		# Move to GPU
		bsz = batch['pixel_values'].shape[0]
		pixel_values = batch['pixel_values'].to(device, non_blocking=True)
		input_ids = batch['input_ids'].to(device, non_blocking=True)

		# Normalize the image
		pixel_values = pixel_values / 255.0
		pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
		pixel_values = pixel_values.to(vision_dtype)

		# Not technically correct but doesn't matter
		attention_mask = torch.ones_like(input_ids)

		# Generate the captions
		generate_ids = llava_model.generate(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, max_new_tokens=256, do_sample=True, suppress_tokens=None, use_cache=True)   # Uses the default which is temp=0.6, top_p=0.9

		# Trim off the prompt
		generate_ids = generate_ids[:, input_ids.shape[1]:]

		# Decode the captions
		captions = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
		captions = (c.replace("<|eot_id|>", "") for c in captions)
		captions = (c.replace("<|finetune_right_pad_id|>", "") for c in captions)
		captions = [c.strip() for c in captions]

		# Debugging
		convo = tokenizer.decode(input_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
		logger.info(f"Convo: {repr(convo.replace('<|reserved_special_token_69|>', ''))}")
		
		for path, caption in zip(batch['paths'], captions):
			cur.execute("UPDATE images SET caption = %s WHERE path = %s", (caption, path))
		
		conn.commit()
		pbar.update(bsz)


class ImageDataset(Dataset):
	def __init__(self, paths: list[str], tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, image_token_id: int, image_seq_length: int):
		self.paths = paths
		self.tokenizer = tokenizer
		self.image_token_id = image_token_id
		self.image_seq_length = image_seq_length
	
	def __len__(self):
		return len(self.paths)
	
	def __getitem__(self, key: tuple[int, str]) -> dict:
		idx, prompt_str = key
		path = self.paths[idx]

		# Preprocess image
		image = Image.open(path)
		image = image.resize((384, 384), Image.LANCZOS)
		image = image.convert("RGB")
		pixel_values = TVF.pil_to_tensor(image)

		# Build the conversation
		convo = [
			{
				"role": "system",
				"content": "You are a helpful image captioner.",
			},
			{
				"role": "user",
				"content": prompt_str,
			},
		]

		# Format the conversation
		convo_string = self.tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = True)
		assert isinstance(convo_string, str)

		# Tokenize the conversation
		convo_tokens = self.tokenizer.encode(convo_string, add_special_tokens=False, truncation=False)

		# Repeat the image tokens
		input_tokens = []
		for token in convo_tokens:
			if token == self.image_token_id:
				input_tokens.extend([self.image_token_id] * self.image_seq_length)
			else:
				input_tokens.append(token)
		
		input_ids = torch.tensor(input_tokens, dtype=torch.long)

		return {
			'path': path,
			'pixel_values': pixel_values,
			'input_ids': input_ids,
		}


def collate_fn(batch: list[dict]) -> dict:
	assert all(item['input_ids'].shape[0] == batch[0]['input_ids'].shape[0] for item in batch), "Expected all items to have the same number of tokens"
	pixel_values = torch.stack([item['pixel_values'] for item in batch])
	paths = [item['path'] for item in batch]

	return {
		'pixel_values': pixel_values,
		'input_ids': torch.stack([item['input_ids'] for item in batch]),
		'paths': paths,
	}


class BatchSamplerWithPrompts(DistributedSampler):
	def __init__(self, dataset: Dataset, num_replicas: Optional[int], rank: Optional[int], batch_size: int) -> None:
		super().__init__(dataset, num_replicas, rank, shuffle=False, seed=0, drop_last=False)
		self.batch_size = batch_size
	
	def __iter__(self) -> Iterator[list[tuple[int, str]]]:
		i = super().__iter__()
		batch = []

		for idx in i:
			batch.append(idx)

			if len(batch) != self.batch_size:
				continue

			prompt = get_random_prompt()

			yield [(idx, prompt) for idx in batch]
			batch = []
		
		if len(batch) > 0:
			prompt = get_random_prompt()
			yield [(idx, prompt) for idx in batch]


def get_random_prompt() -> str:
	# 20% training prompt
	# 10% MidJourney
	# 60% Descriptive
	# 10% Descriptive (Informal)
	caption_type = random.choices(["Descriptive", "Descriptive (Informal)", "Training Prompt", "MidJourney"], weights=[6, 1, 2, 1])[0]

	# Word count
	# Rounded up to the nearest 10
	word_count = int(np.random.normal(MEAN_WORDS, STD_WORDS))
	word_count = 10 * ((word_count + 9) // 10)
	word_count = min(max(word_count, 20), 170)

	prompt = CAPTION_TYPE_MAP[caption_type][1]
	assert "{word_count}" in prompt, f"Expected {prompt} to contain {word_count}"

	if random.random() < 0.25:
		prompt += " Include whether the image is sfw, suggestive, or nsfw."
	
	prompt = prompt.format(word_count=word_count)
	assert "{" not in prompt, f"Expected all placeholders to be filled, got {prompt}"

	return prompt


def distributed_rank():
	if not torch.distributed.is_initialized():
		return 0
	
	return torch.distributed.get_rank()


def distributed_world_size():
	if not torch.distributed.is_initialized():
		return 1
	
	return torch.distributed.get_world_size()


def distributed_setup():
	torch.distributed.init_process_group(backend="nccl", init_method="env://")


def distributed_cleanup():
	torch.distributed.destroy_process_group()


def print_rank_0(*args, **kwargs):
	if distributed_rank() == 0:
		print(*args, **kwargs)


def log_rank_0(logger: logging.Logger, level: int, *args, **kwargs):
	if distributed_rank() == 0:
		logger.log(level, *args, **kwargs)


if __name__ == "__main__":
	distributed_setup()
	torch.cuda.set_device(distributed_rank())
	main()
	distributed_cleanup()