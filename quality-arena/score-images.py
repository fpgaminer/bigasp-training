#!/usr/bin/env python3
"""
Score all the images using our scoring model.
"""
import torch
from tqdm import tqdm
import PIL.Image
import torch.utils.data
import sqlite3
import argparse
from models import ScoreClassifier
from typing import Iterable, Iterator, TypeVar
import itertools
from pathlib import Path


PIL.Image.MAX_IMAGE_PIXELS = 933120000


parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--database", type=str, default="../data/clip-embeddings.sqlite3")


@torch.no_grad()
def main():
	args = parser.parse_args()

	# Load model
	model = ScoreClassifier(768, 0.0, 10)
	model.load_state_dict(torch.load(Path(__file__).resolve().parent / "scorer.pt"))
	model.requires_grad_(False)
	model.eval()
	model = model.to("cuda")

	# Connect to the database
	conn = sqlite3.connect(args.database)
	cur = conn.cursor()

	# Fetch a list of all paths we need to work with
	cur.execute("SELECT path FROM images WHERE embedding IS NOT NULL AND score IS NULL")
	paths = [path for path, in cur.fetchall()]

	print(f"Found {len(paths)} paths to process")

	pbar = tqdm(total=len(paths), desc="Scoring images...", dynamic_ncols=True)
	for batch in batcher(paths, args.batch_size):
		# Fetch embeddings
		cur.execute("SELECT path, embedding FROM images WHERE path IN ({})".format(",".join("?" for _ in batch)), batch)
		paths, embeddings = zip(*cur.fetchall())
		embeddings = [torch.frombuffer(e, dtype=torch.float16) for e in embeddings]
		embeddings = torch.stack(embeddings).to(torch.float32).to("cuda")

		# Run through the model
		scores = model(embeddings)
		assert scores.shape == (len(embeddings), 10)
		scores = torch.softmax(scores, dim=1)
		scores = torch.argmax(scores, dim=1)

		# Insert scores into the database
		for path, score in zip(paths, scores):
			cur.execute("UPDATE images SET score = ? WHERE path = ?", (int(score), path))
		
		conn.commit()
		pbar.update(len(paths))


T = TypeVar("T")

def batcher(iterable: Iterable[T], n: int) -> Iterator[list[T]]:
	iterator = iter(iterable)
	while batch := list(itertools.islice(iterator, n)):
		yield batch


if __name__ == "__main__":
	main()