#!/usr/bin/env python3
"""
Embed all the images using CLIP.
Useful for later filtering the images based on their embeddings.
"""
from transformers import CLIPProcessor, CLIPModel
import torch
from tqdm import tqdm
from PIL import Image
import PIL.Image
import safetensors
import torch.utils.data
import safetensors.torch
from torch.utils.data import Dataset
import argparse
import psycopg
from pathlib import Path


PIL.Image.MAX_IMAGE_PIXELS = 933120000


parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, default=32)


@torch.no_grad()
def main():
	args = parser.parse_args()

	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True

	# Load CLIP
	clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
	assert isinstance(clip_model, CLIPModel)
	clip_model = clip_model.vision_model.to("cuda")
	clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
	assert isinstance(clip_processor, CLIPProcessor)

	# Connect to the database
	#conn = sqlite3.connect(args.database)
	conn = psycopg.connect(dbname='postgres', user='postgres', host=str(Path.cwd() / "pg-socket"))
	cur = conn.cursor()

	# Get a list of remaining paths to process
	cur.execute("SELECT path FROM images WHERE embedding IS NULL")
	paths = [path for path, in cur.fetchall()]

	print(f"Found {len(paths)} paths to process")

	dataloader = torch.utils.data.DataLoader(ImageDataset(paths, clip_processor), batch_size=128, num_workers=args.num_workers, shuffle=False, drop_last=False, prefetch_factor=4)

	clip_model.requires_grad_(False)
	clip_model.eval()
	clip_model = torch.compile(clip_model)

	pbar = tqdm(total=len(paths), desc="Embedding images...", dynamic_ncols=True)
	for images, paths in dataloader:
		# Run through CLIP
		outputs = clip_model(pixel_values=images.to('cuda'), output_hidden_states=True)
		
		# Grab the last layer outputs before normalization
		# We also don't use the vision projection layer
		# All those can be quickly applied as/if needed when using the dataset
		embedding = outputs.hidden_states[-1][:, 0, :].detach().cpu().to(torch.float16)

		for path, embedding in zip(paths, embedding):
			if path == "":
				continue

			b = safetensors.torch._tobytes(embedding, "embedding")
			assert len(b) == clip_model.config.hidden_size * 2
			cur.execute("UPDATE images SET embedding = %s WHERE path = %s", (b, str(path)))
		
		conn.commit()

		pbar.update(len(paths))


class ImageDataset(Dataset):
	def __init__(self, paths: list[str], processor: CLIPProcessor):
		self.paths = paths
		self.processor = processor
	
	def __len__(self):
		return len(self.paths)
	
	def __getitem__(self, index):
		path = self.paths[index]
		try:
			image = Image.open(path)
			image = self.processor(images=image, return_tensors="pt")["pixel_values"]
		except Exception as e:
			print(f"Failed to open image {path}: {e}")
			return torch.empty(3, 224, 224), ""

		return image.squeeze(0), path


if __name__ == "__main__":
	main()