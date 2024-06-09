#!/usr/bin/env python3
"""
Pre-encode all the images into latents using the VAE.
Uses both a float32 and a bfloat16 version of the VAE, to help make
the final model more robust to end users using bfloat16.
"""
import itertools
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL
from accelerate import Accelerator
import torch
import argparse
import random
import torchvision.transforms.functional as TVF
from tqdm import tqdm
import safetensors.torch
import struct
import sqlite3
import gzip


parser = argparse.ArgumentParser()
parser.add_argument("--base-model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
parser.add_argument("--base-revision", type=str, default="462165984030d82259a11f4367a4eed129e94a7b")
parser.add_argument("--device-batch-size", type=int, default=1)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--output-path", type=str, default="data/vaes")
parser.add_argument("--database", type=str, default="data/clip-embeddings.sqlite3")
parser.add_argument("--mixed-vaes", action="store_true", help="Encode with different VAEs to improve robustness")


# This mostly matches what is written in the SDXL paper
# Except we filter out more extreme aspect ratios
# And the paper didn't include 1344x704 for some reason?
AR_BUCKETS = list(range(512, 2049, 64))
AR_BUCKETS = itertools.product(AR_BUCKETS, AR_BUCKETS)
AR_BUCKETS = set([v for v in AR_BUCKETS if v[0] * v[1] <= 1024*1024 and v[0] * v[1] >= 946176 and v[0]/v[1] >= 0.5 and v[0]/v[1] <= 3.0])


@torch.no_grad()
def main():
	args = parser.parse_args()
	output_path = Path(args.output_path)
	accelerator = Accelerator()
	
	# Load models
	vaes = []

	# Standard VAE at float32
	vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", revision=args.base_revision, torch_dtype=torch.float32, use_safetensors=True)
	assert isinstance(vae, AutoencoderKL)
	print(f"VAE scale: {vae.config.scaling_factor}")
	vaes.append((vae, torch.float32))

	if args.mixed_vaes:
		# Standard VAE at loaded from float16, but running at float32
		vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", revision=args.base_revision, torch_dtype=torch.float16, use_safetensors=True)
		assert isinstance(vae, AutoencoderKL)
		vaes.append((vae.to(torch.float32), torch.float32))

		# Standard VAE at bfloat16
		vae_bf16 = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", revision=args.base_revision, torch_dtype=torch.bfloat16, use_safetensors=True)
		assert isinstance(vae_bf16, AutoencoderKL)
		vaes.append((vae_bf16, torch.bfloat16))

		# fp16-fix VAE
		vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
		assert isinstance(vae, AutoencoderKL)
		vaes.append((vae, torch.float16))

	for vae,_ in vaes:
		vae.requires_grad_(False)
		vae.eval()

	# Connect to the database
	conn = sqlite3.connect(args.database)
	cur = conn.cursor()

	# Fetch a list of all paths we need to work on
	cur.execute("SELECT id, path FROM images WHERE embedding IS NOT NULL AND score IS NOT NULL AND score > 0")
	image_paths = [(id, path) for id,path in cur.fetchall()]

	# Filter out images we've already processed
	print(f"{len(image_paths)} images to process")
	image_paths = [p for p in image_paths if not encoded_path(p[0], output_path).exists()]
	print(f"{len(image_paths)} images to process after filtering")
	
	dataset = ImageDataset(image_paths)

	dataloader = DataLoader(
		dataset,
		batch_size=args.device_batch_size,
		num_workers=args.num_workers,
		pin_memory=True,
		drop_last=False,
		shuffle=False,
		pin_memory_device=accelerator.device,
	)

	# Compile
	vaes = [(torch.compile(vae), dtype) for vae, dtype in vaes]

	# Accelerate
	dataloader = accelerator.prepare(dataloader)
	vaes = [(accelerator.prepare(vae), dtype) for vae, dtype in vaes]

	# Encode
	for batch in tqdm(dataloader, "Encoding...", disable=not accelerator.is_local_main_process, dynamic_ncols=True):
		images = batch['images'].to(accelerator.device)
		original_widths = batch['original_width']
		original_heights = batch['original_height']
		crop_xs = batch['crop_x']
		crop_ys = batch['crop_y']
		indexes = batch['index']

		# Encode
		vae, vae_dtype = random.choice(vaes)
		latents = vae.encode(images.to(vae_dtype)).latent_dist.sample()
		latents = latents * vae.config.scaling_factor

		# Convert latents to float16, move to CPU, and check for NaNs
		latents = latents.to(dtype=torch.float16, device='cpu')
		assert torch.isfinite(latents).all()

		# Save
		for latent, index, original_width, original_height, crop_x, crop_y in zip(latents, indexes, original_widths, original_heights, crop_xs, crop_ys):
			encoded_path_i = encoded_path(index.item(), output_path)
			encoded_path_i.parent.mkdir(parents=True, exist_ok=True)
			tmppath = encoded_path_i.with_suffix(".tmp")

			latent_bytes = safetensors.torch._tobytes(latent, "latent")
			assert latent.shape[0] == 4, f"Expected 4 channels, got {latent.shape[0]}"
			assert len(latent_bytes) == latent.shape[1] * latent.shape[2] * 4 * 2, f"Expected {latent.shape[1]}x{latent.shape[2]}x4x2 bytes, got {len(latent_bytes)} bytes"
			metadata = struct.pack("<IIIIIII", index, original_width, original_height, crop_x, crop_y, latent.shape[1], latent.shape[2])

			with gzip.open(tmppath, "wb") as f:
			#with open(tmppath, "wb") as f:
				f.write(metadata)
				f.write(latent_bytes)
			
			tmppath.rename(encoded_path_i)


def encoded_path(index: int, output_path: Path):
	dir = index % 1000
	return output_path / f"{dir:03d}" / f"{index}.bin.gz"


class ImageDataset(Dataset):
	def __init__(self, image_paths: list[tuple[int, Path]]):
		self.image_paths = image_paths
	
	def __len__(self):
		return len(self.image_paths)
	
	def __getitem__(self, idx):
		index, image_path = self.image_paths[idx]
		image = Image.open(image_path).convert("RGB")
		original_size = image.size
		ar = image.width / image.height

		# Find the AR bucket that is closest to the image's aspect ratio
		ar_bucket = min(AR_BUCKETS, key=lambda v: abs(v[0]/v[1] - ar))

		# Scale the image
		scale = max(ar_bucket[0] / image.width, ar_bucket[1] / image.height)
		image = image.resize((int(image.width * scale + 0.5), int(image.height * scale + 0.5)), Image.LANCZOS)
		assert image.width == ar_bucket[0] or image.height == ar_bucket[1]
		assert image.width >= ar_bucket[0] and image.height >= ar_bucket[1]

		# Random crop
		# Paste onto a new image to avoid some edge cases I've encountered
		crop_x = random.randint(0, image.width - ar_bucket[0])
		crop_y = random.randint(0, image.height - ar_bucket[1])
		cropped = Image.new("RGB", (ar_bucket[0], ar_bucket[1]))
		cropped.paste(image, (-crop_x, -crop_y))

		# Convert to tensor
		image_tensor = TVF.pil_to_tensor(cropped)
		image_tensor = image_tensor / 255.0  # 0-1
		image_tensor = image_tensor - 0.5  # -0.5 to 0.5
		image_tensor = image_tensor * 2.0 # -1 to 1

		# N.B. The algorithm outlined in the SDXL paper indicates that crop_x and crop_y are expressed in the resized image's coordinate system.
		# So they represent, as they do here, the number of pixels cropped from the left and top of the resized image.
		return {
			'images': image_tensor,
			'original_width': original_size[0],
			'original_height': original_size[1],
			'crop_x': crop_x,
			'crop_y': crop_y,
			'index': index,
		}


if __name__ == "__main__":
	main()