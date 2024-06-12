#!/usr/bin/env python3
"""
Go through all the images in the database and set the `watermark` column if the image has a watermark.
"""
from PIL import Image
import sqlite3
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import torch
import argparse
import PIL.Image
from tqdm import tqdm
import torch.utils.data
import safetensors.torch
import torchvision.transforms.functional as TVF


PIL.Image.MAX_IMAGE_PIXELS = 933120000


parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--database", type=str, default="../data/clip-embeddings.sqlite3")


@torch.no_grad()
def main():
	args = parser.parse_args()

	# Load model
	processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
	assert isinstance(processor, Owlv2Processor)
	model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
	assert isinstance(model, Owlv2ForObjectDetection)
	model.eval()
	model = model.to("cuda") # type: ignore
	model = torch.compile(model)

	# Connect to the database
	conn = sqlite3.connect(args.database)
	cur = conn.cursor()

	# Fetch a list of all paths we need to work with
	cur.execute("SELECT path FROM images WHERE embedding IS NOT NULL AND watermark IS NULL")
	paths = [path for path, in cur.fetchall()]
	dataset = ImageDataset(paths, processor)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=12, shuffle=False, pin_memory=True, drop_last=False)

	print(f"Found {len(paths)} paths to process")

	texts = [["a watermark"] for _ in range(args.batch_size)]
	inputs = processor(text=texts, return_tensors="pt")
	inputs = inputs.to("cuda")

	pbar = tqdm(total=len(paths), desc="Analyzing images...", dynamic_ncols=True, smoothing=0.01)
	for paths, pixel_values, image_size in dataloader:
		pixel_values = pixel_values.to("cuda")
		with torch.cuda.amp.autocast():
			outputs = model(pixel_values=pixel_values, attention_mask=inputs['attention_mask'][:len(paths)], input_ids=inputs['input_ids'][:len(paths)])

		results = processor.post_process_object_detection(outputs=outputs, target_sizes=None, threshold=0.1)

		for path, result in zip(paths, results):
			if len(result['boxes']) > 0:
				# Encode the boxes into a binary format
				packed_boxes = safetensors.torch._tobytes(result['boxes'], "boxes")
				assert len(packed_boxes) == len(result['boxes']) * 8, f"Expected {len(result['boxes']) * 8} bytes, got {len(packed_boxes)} bytes"
				cur.execute("UPDATE images SET watermark = 1, watermark_boxes = ? WHERE path = ?", (packed_boxes, path))
			else:
				cur.execute("UPDATE images SET watermark = 0 WHERE path = ?", (path,))
		
		pbar.update(len(paths))
		conn.commit()


IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]


class ImageDataset(torch.utils.data.Dataset):
	def __init__(self, paths: list[str], processor: Owlv2Processor):
		self.paths = paths
		self.processor = processor

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		path = self.paths[index]
		image = Image.open(path).convert("RGB")
		#inputs = self.processor(images=image, return_tensors="pt")   # This is the normal, and most accurate way to do it, but _very_ slow

		# Pad to square
		big_side = max(image.size)
		new_image = Image.new("RGB", (big_side, big_side), (128, 128, 128))
		new_image.paste(image, (0, 0))

		# Resize to 960x960
		preped = new_image.resize((960, 960), Image.BICUBIC)  # Bicubic performed best in my tests (even compared to Lanczos)

		# Convert to tensor and normalize
		preped = TVF.pil_to_tensor(preped)
		preped = preped / 255.0
		preped = TVF.normalize(preped, IMAGE_MEAN, IMAGE_STD)

		return path, preped, image.size


if __name__ == "__main__":
	main()