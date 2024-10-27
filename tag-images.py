#!/usr/bin/env python3
"""
Use JoyTag to tag all the images.
"""
import torch
from tqdm import tqdm
from PIL import Image
import PIL.Image
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import argparse
from JoyTag import VisionModel
import torchvision.transforms.functional as TVF
from pathlib import Path
import torch.amp
import psycopg
from torch import multiprocessing


PIL.Image.MAX_IMAGE_PIXELS = 933120000
JOYTAG_PATH = "joytag"
THRESHOLD = 0.4


parser = argparse.ArgumentParser()
parser.add_argument("--num-workers", type=int, default=32)
parser.add_argument("--batch-size", type=int, default=256)


@torch.no_grad()
def main():
	args = parser.parse_args()

	# Load JoyTag
	model = VisionModel.load_model(JOYTAG_PATH, device='cuda')
	model.eval()
	model = torch.compile(model)

	with open(Path(JOYTAG_PATH) / "top_tags.txt", "r") as f:
		top_tags = [line.strip() for line in f.readlines() if line.strip()]

	# Connect to the database
	conn = psycopg.connect(dbname='postgres', user='postgres', host=str(Path.cwd() / "pg-socket"))
	cur = conn.cursor()

	# Fetch a list of all paths we need to work on
	cur.execute("SELECT path FROM images WHERE embedding IS NOT NULL AND tag_string IS NULL")
	paths = [path for path, in cur.fetchall()]

	print(f"Found {len(paths)} paths to process")

	dataloader = DataLoader(ImageDataset(paths, model.image_size), batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, drop_last=False, prefetch_factor=4)

	# Result writer
	#result_queue = multiprocessing.Queue()
	#result_process = multiprocessing.Process(target=result_worker, args=(result_queue,))
	#result_process.start()

	pbar = tqdm(total=len(paths), desc="Tagging images...", dynamic_ncols=True)
	for images, paths in dataloader:
		# Move to GPU
		images = images.to('cuda', non_blocking=True)

		# Put into range [0-1]
		images = images / 255.0

		# Normalize
		images = TVF.normalize(images, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

		with torch.amp.autocast_mode.autocast('cuda', enabled=True):
			preds = model({'image': images})
			#preds = preds['tags'].sigmoid().cpu()
			preds = preds['tags'].sigmoid() > THRESHOLD
			assert preds.shape == (len(paths), len(top_tags))
			preds = preds.cpu()
		
		for path, pred in zip(paths, preds):
			pred = pred.nonzero(as_tuple=True)[0]
			tag_string = ",".join(top_tags[i] for i in pred)

			cur.execute("UPDATE images SET tag_string = %s WHERE path = %s", (tag_string, path))
		
		conn.commit()
		
		#result_queue.put((paths, preds))
		
		#assert preds.shape == (len(images), len(top_tags))

		#for path, pred in zip(paths, preds):
		#	predicted_tags = [tag for tag,score in zip(top_tags, pred) if score > THRESHOLD]
		#	tag_string = ",".join(predicted_tags)
		
		#	cur.execute("UPDATE images SET tag_string = %s WHERE path = %s", (tag_string, path))
		
		#conn.commit()
		pbar.update(len(images))
	
	#result_queue.put(None)
	#result_process.join()


class ImageDataset(Dataset):
	def __init__(self, paths: list[str], size: int):
		self.paths = paths
		self.size = size
	
	def __len__(self):
		return len(self.paths)
	
	def __getitem__(self, idx):
		path = self.paths[idx]
		image_tensor = prepare_image(Image.open(path), self.size)

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
	image_tensor = TVF.pil_to_tensor(padded_image)# / 255.0

	# Normalize
	#image_tensor = TVF.normalize(image_tensor, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

	return image_tensor


def result_worker(queue: multiprocessing.Queue):
	# Connect to the database
	conn = psycopg.connect(dbname='postgres', user='postgres', host=str(Path.cwd() / "pg-socket"))
	cur = conn.cursor()

	# Read top tags
	with open(Path(JOYTAG_PATH) / "top_tags.txt", "r") as f:
		top_tags = [line.strip() for line in f.readlines() if line.strip()]

	# Process results
	while True:
		data = queue.get()

		if data is None:
			break

		paths, preds = data

		for path, pred in zip(paths, preds):
			tag_string = ",".join(tag for tag,score in zip(top_tags, pred) if score)
		
			cur.execute("UPDATE images SET tag_string = %s WHERE path = %s", (tag_string, path))
		
		conn.commit()
	
	conn.close()


if __name__ == "__main__":
	main()