#!/usr/bin/env python3
"""
Use JoyTag to tag all the images.
"""
import torch
from tqdm import tqdm
from PIL import Image
import PIL.Image
import torch.utils.data
from torch.utils.data import Dataset
import sqlite3
import argparse
from JoyTag import VisionModel
import torchvision.transforms.functional as TVF
from pathlib import Path
import torch.amp


PIL.Image.MAX_IMAGE_PIXELS = 933120000
JOYTAG_PATH = "joytag"
THRESHOLD = 0.4


parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--batch-size", type=int, default=128)


@torch.no_grad()
def main():
	args = parser.parse_args()

	# Load JoyTag
	model = VisionModel.load_model(JOYTAG_PATH, device='cuda')
	model.eval()

	with open(Path(JOYTAG_PATH) / "top_tags.txt", "r") as f:
		top_tags = [line.strip() for line in f.readlines() if line.strip()]

	# Connect to the database
	conn = sqlite3.connect("data/clip-embeddings.sqlite3")
	cur = conn.cursor()

	# Fetch a list of all paths we need to work on
	cur.execute("SELECT path FROM images WHERE embedding IS NOT NULL AND tag_string IS NULL")
	paths = [path for path, in cur.fetchall()]

	print(f"Found {len(paths)} paths to process")

	dataloader = torch.utils.data.DataLoader(ImageDataset(paths, model.image_size), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False)

	pbar = tqdm(total=len(paths), desc="Tagging images...", dynamic_ncols=True)
	for images, paths in dataloader:
		# Move images to GPU
		images = images.to('cuda')

		with torch.amp.autocast_mode.autocast('cuda', enabled=True):
			preds = model({'image': images})
			preds = preds['tags'].sigmoid().cpu()
		
		assert preds.shape == (len(images), len(top_tags))
		
		for path, pred in zip(paths, preds):
			predicted_tags = [tag for tag,score in zip(top_tags, pred) if score > THRESHOLD]
			tag_string = ",".join(predicted_tags)
		
			cur.execute("UPDATE images SET tag_string = ? WHERE path = ?", (tag_string, str(path)))
		conn.commit()

		pbar.update(len(paths))


class ImageDataset(Dataset):
	def __init__(self, paths: list[str], target_size: int):
		self.paths = paths
		self.target_size = target_size
	
	def __len__(self):
		return len(self.paths)
	
	def __getitem__(self, index):
		path = self.paths[index]
		image_tensor = prepare_image(Image.open(path), self.target_size)

		return image_tensor, path


def prepare_image(image: Image.Image, target_size: int) -> torch.Tensor:
	# Pad image to square
	image_shape = image.size
	max_dim = max(image_shape)
	pad_left = (max_dim - image_shape[0]) // 2
	pad_top = (max_dim - image_shape[1]) // 2

	padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
	padded_image.paste(image, (pad_left, pad_top))

	# Resize image
	if max_dim != target_size:
		padded_image = padded_image.resize((target_size, target_size), Image.BICUBIC)
	
	# Convert to tensor
	image_tensor = TVF.pil_to_tensor(padded_image) / 255.0

	# Normalize
	image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

	return image_tensor


if __name__ == "__main__":
	main()