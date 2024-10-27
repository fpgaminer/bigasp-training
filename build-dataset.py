#!/usr/bin/env python3
"""
Build the parquet dataset that training will use.
This uses the data from the database and the precomputed latents.
"""
import pyarrow as pa
import pyarrow.parquet as pq
import math
from tqdm import tqdm
from pathlib import Path
import gzip
import struct
from typing import Iterable, Iterator, TypeVar
import itertools
import argparse
import psycopg
from hashlib import md5
from transformers import CLIPTokenizer
import multiprocessing


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="data/dataset.parquet")
parser.add_argument("--vae-path", type=str, default="data/vaes")


VALID_SOURCES = {"fansly", "flickr", "onlyfans", "unsplash"}


schema = pa.schema([
	pa.field("image_hash", pa.string()),   # Uniquely identifies the image
	pa.field("tag_string", pa.string()),
	pa.field("caption", pa.string()),
	pa.field("score", pa.int32()),
	pa.field("latent_width", pa.int32()),
	pa.field("latent_height", pa.int32()),
	pa.field("n_tokens", pa.int32()),
])


def main():
	args = parser.parse_args()

	# Load the tokenizer
	base_model = "stabilityai/stable-diffusion-xl-base-1.0"
	base_revision = "462165984030d82259a11f4367a4eed129e94a7b"
	tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", revision=base_revision, use_fast=True)
	assert isinstance(tokenizer, CLIPTokenizer)

	# Connect to the database
	conn = psycopg.connect(dbname='postgres', user='postgres', host=str(Path.cwd() / "pg-socket"))

	# Fetch all records
	# Images with a score of 0 are discarded
	print("Fetching records...")
	with conn.cursor('dataset-builder') as cur:
		cur.execute("SELECT path, tag_string, score, subreddit, caption, caption_2, caption_3, caption_4, watermark, source FROM images WHERE embedding IS NOT NULL AND score IS NOT NULL AND score > 0 AND tag_string IS NOT NULL and caption IS NOT NULL")
		records = []

		for path, tag_string, score, subreddit, caption, caption_2, caption_3, caption_4, watermark, source in tqdm(cur, desc="Reading records", dynamic_ncols=True):
			image_hash = md5(path.encode()).hexdigest()
			caption = caption
			if caption_2 is not None:
				caption = caption_2
			if caption_3 is not None:
				caption = caption_3
			if caption_4 is not None:
				caption = caption_4
			
			records.append({
				'image_hash': image_hash,
				'path': path,
				'tag_string': tag_string,
				'score': score,
				'subreddit': subreddit,
				'caption': caption,
				'watermark': watermark,
				'source': source,
			})

	# Read latent sizes
	with multiprocessing.Pool(processes=8, initializer=worker_init, initargs=(args, tokenizer)) as pool:
		records = list(tqdm(pool.imap_unordered(latent_worker, records), total=len(records), desc="Reading latent sizes", dynamic_ncols=True))
	
	# for record in tqdm(records, desc="Reading latent sizes", dynamic_ncols=True):
	# 	precomputed_path = Path(args.vae_path) / record['image_hash'][:2] / record['image_hash'][2:4] / f"{record['image_hash']}.bin.gz"
	# 	if not precomputed_path.exists():
	# 		print(f"Missing precomputed file for image_hash {record['image_hash']}, {precomputed_path} - skipping")
	# 		continue

	# 	with gzip.open(precomputed_path, "rb") as f:
	# 		original_width, original_height, crop_x, crop_y, latent_width, latent_height = struct.unpack("<IIIIII", f.read(24))
	# 		record['latent_width'] = latent_width
	# 		record['latent_height'] = latent_height
	
	# Remove records with missing latents
	print(f"Records before latent filter: {len(records)}")
	records = [record for record in records if 'latent_width' in record]
	print(f"Records after: {len(records)}")
	
	# Append subreddit, watermark, source to tag_string
	for record in records:
		if record['subreddit'] is not None and record['subreddit'] != "":
			record['tag_string'] += f",{record['subreddit']},reddit"
		
		if record['watermark'] is not None and record['watermark'] == 1:
			record['tag_string'] += ",watermark"
		
		if record['source'] is not None and record['source'] != "" and record['source'] in VALID_SOURCES:
			record['tag_string'] += f",{record['source']}"
		
	dataset_writer(args.output, records)


def worker_init(args: argparse.Namespace, tokenizer: CLIPTokenizer):
	global g_args, g_tokenizer
	g_args = args
	g_tokenizer = tokenizer


def latent_worker(record: dict):
	assert g_args is not None and g_tokenizer is not None

	caption_tokens = g_tokenizer.encode(record['caption'], add_special_tokens=False, truncation=False, padding=False)
	record['n_tokens'] = len(caption_tokens)

	precomputed_path = Path(g_args.vae_path) / record['image_hash'][:2] / record['image_hash'][2:4] / f"{record['image_hash']}.bin.gz"
	if not precomputed_path.exists():
		print(f"Missing precomputed file for image_hash {record['image_hash']}, {precomputed_path} - skipping")
		return record

	with gzip.open(precomputed_path, "rb") as f:
		original_width, original_height, crop_x, crop_y, latent_width, latent_height = struct.unpack("<IIIIII", f.read(24))
		record['latent_width'] = latent_width
		record['latent_height'] = latent_height
	
	return record


def dataset_writer(dest_path: Path | str, records: list):
	with pq.ParquetWriter(dest_path, schema) as writer:
		for batch in tqdm(batcher(records, 1000), total=math.ceil(len(records) / 1000), dynamic_ncols=True):
			hashes = [row['image_hash'] for row in batch]
			tag_strings = [row['tag_string'] for row in batch]
			captions = [row['caption'] for row in batch]
			scores = [int(row['score']) for row in batch]
			latent_widths = [int(row['latent_width']) for row in batch]
			latent_heights = [int(row['latent_height']) for row in batch]
			n_tokens = [int(row['n_tokens']) for row in batch]

			batch = pa.RecordBatch.from_arrays([
				pa.array(hashes, type=pa.string()),
				pa.array(tag_strings, type=pa.string()),
				pa.array(captions, type=pa.string()),
				pa.array(scores, type=pa.int32()),
				pa.array(latent_widths, type=pa.int32()),
				pa.array(latent_heights, type=pa.int32()),
				pa.array(n_tokens, type=pa.int32()),
			], schema=schema)
			writer.write(batch)


T = TypeVar("T")
def batcher(iterable: Iterable[T], n: int) -> Iterator[list[T]]:
	iterator = iter(iterable)
	while batch := list(itertools.islice(iterator, n)):
		yield batch


if __name__ == "__main__":
	main()