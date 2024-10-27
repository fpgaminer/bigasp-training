#!/usr/bin/env python3
"""
Pre-encode all the images into latents using the VAE.
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
import gzip
import psycopg
from hashlib import md5
import PIL.Image
from torch import multiprocessing
import dataclasses


PIL.Image.MAX_IMAGE_PIXELS = 933120000
g_semaphore = None


parser = argparse.ArgumentParser()
parser.add_argument("--base-model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
parser.add_argument("--base-revision", type=str, default="462165984030d82259a11f4367a4eed129e94a7b")
parser.add_argument("--device-batch-size", type=int, default=1)
parser.add_argument("--num-workers", type=int, default=8)
parser.add_argument("--output-path", type=str, default="data/vaes")
parser.add_argument("--hash-filter", type=str, default=None)   # Dumb way to handle multi-GPU for this task, but whatever


# This mostly matches what is written in the SDXL paper
# Except we filter out more extreme aspect ratios
# And the paper didn't include 1344x704 for some reason?
_AR_BUCKETS = list(range(512, 2049, 64))
_AR_BUCKETS = itertools.product(_AR_BUCKETS, _AR_BUCKETS)
AR_BUCKETS: set[tuple[int, int]] = set([v for v in _AR_BUCKETS if v[0] * v[1] <= 1024*1024 and v[0] * v[1] >= 946176 and v[0]/v[1] >= 0.333 and v[0]/v[1] <= 3.0])


@dataclasses.dataclass
class Batch:
	images: torch.Tensor
	original_widths: list[int]
	original_heights: list[int]
	crop_xs: list[int]
	crop_ys: list[int]
	image_hashes: list[str]


@dataclasses.dataclass
class Result:
	latents: torch.Tensor
	original_widths: list[int]
	original_heights: list[int]
	crop_xs: list[int]
	crop_ys: list[int]
	image_hashes: list[str]


@torch.no_grad()
def main():
	args = parser.parse_args()
	output_path = Path(args.output_path)

	torch.backends.cuda.matmul.allow_tf32 = True
	torch.backends.cudnn.allow_tf32 = True
	
	# Standard VAE at float32
	vae = AutoencoderKL.from_pretrained(args.base_model, subfolder="vae", revision=args.base_revision, torch_dtype=torch.float32, use_safetensors=True)
	assert isinstance(vae, AutoencoderKL)
	vae.eval()
	print(f"VAE scale: {vae.config.scaling_factor}")
	vae.to('cuda')
	vae = torch.compile(vae)

	# Connect to the database and fetch a list of all paths we need to work on
	with psycopg.connect(dbname='postgres', user='postgres', host=str(Path.cwd() / "pg-socket")) as conn:
		cur = conn.cursor()

		cur.execute("SELECT path FROM images WHERE embedding IS NOT NULL AND score IS NOT NULL AND score > 0")
		image_paths = [(md5(path.encode()).hexdigest(), Path(path)) for path, in cur.fetchall()]

	# Filter out images we've already processed
	print(f"{len(image_paths)} images to process")
	image_paths = [p for p in image_paths if not encoded_path(p[0], output_path).exists()]
	print(f"{len(image_paths)} images to process after filtering")

	# Filter out images that don't match the hash filter
	if args.hash_filter is not None and args.hash_filter != "":
		image_paths = [p for p in image_paths if p[0][0] in args.hash_filter]
		print(f"{len(image_paths)} images to process after hash filtering")

	# Result writer
	result_queue = multiprocessing.Queue()
	result_process = multiprocessing.Process(target=result_worker, args=(result_queue, output_path))
	result_process.start()

	pbar = tqdm(total=len(image_paths), desc="Encoding images...", dynamic_ncols=True, smoothing=0.01)
	remaining = len(image_paths)
	semaphore_1 = multiprocessing.Semaphore(args.num_workers * 4)
	batches: dict[tuple[int, int], Batch] = {}

	for bucket in AR_BUCKETS:
		batches[bucket] = Batch(
			images=torch.empty((args.device_batch_size, 3, bucket[1], bucket[0]), dtype=torch.uint8, device='cuda'),
			original_widths=[],
			original_heights=[],
			crop_xs=[],
			crop_ys=[],
			image_hashes=[],
		)
	
	with multiprocessing.Pool(args.num_workers, initializer=init_worker, initargs=(semaphore_1,)) as pool:
		images = pool.imap_unordered(worker, image_paths)

		while True:
			# Get an image from our worker pool
			image = next(images, None)

			if image is not None:
				pbar.update(1)
				remaining -= 1
				semaphore_1.release()

				# Add to our batch buckets
				k = (image['image'].size(2), image['image'].size(1))
				i = len(batches[k].image_hashes)
				batches[k].images[i] = image['image'].to('cuda', non_blocking=True)
				batches[k].original_widths.append(image['original_width'])
				batches[k].original_heights.append(image['original_height'])
				batches[k].crop_xs.append(image['crop_x'])
				batches[k].crop_ys.append(image['crop_y'])
				batches[k].image_hashes.append(image['image_hash'])
			
			# If we have enough images, run the model
			batch = None

			for k, v in batches.items():
				if len(v.image_hashes) >= args.device_batch_size or (remaining == 0 and len(v.image_hashes) > 0):
					batch = v
					break
			
			if batch is None and remaining == 0:
				break

			if batch is None:
				continue

			# Normalize
			pixel_values = batch.images / 255.0  # 0-1
			pixel_values = pixel_values - 0.5    # -0.5 to 0.5
			pixel_values = pixel_values * 2.0    # -1 to 1

			# Encode
			latents = vae.encode(pixel_values.to(torch.float32)).latent_dist.sample()
			latents = latents * vae.config.scaling_factor

			# Convert latents to float16, move to CPU, and check for NaNs
			latents = latents.to(dtype=torch.float16, device='cpu')
			assert torch.isfinite(latents).all()

			result_queue.put(Result(
				latents=latents,
				original_widths=batch.original_widths,
				original_heights=batch.original_heights,
				crop_xs=batch.crop_xs,
				crop_ys=batch.crop_ys,
				image_hashes=batch.image_hashes,
			))

			# Clear the batch
			batch.original_widths = []
			batch.original_heights = []
			batch.crop_xs = []
			batch.crop_ys = []
			batch.image_hashes = []
	
	# Clean up and wait for the result writer to finish
	result_queue.put(None)
	result_process.join()


def result_worker(queue: multiprocessing.Queue, output_path: Path):
	while True:
		result = queue.get()

		if result is None:
			break

		for i in range(len(result.image_hashes)):
			latent = result.latents[i]
			original_width = result.original_widths[i]
			original_height = result.original_heights[i]
			crop_x = result.crop_xs[i]
			crop_y = result.crop_ys[i]
			image_hash = result.image_hashes[i]

			encoded_path_i = encoded_path(image_hash, output_path)
			encoded_path_i.parent.mkdir(parents=True, exist_ok=True)
			tmppath = encoded_path_i.with_suffix(".tmp")

			latent_bytes = safetensors.torch._tobytes(latent, "latent")
			assert latent.shape[0] == 4, f"Expected 4 channels, got {latent.shape[0]}"
			assert len(latent_bytes) == latent.shape[1] * latent.shape[2] * 4 * 2, f"Expected {latent.shape[1]}x{latent.shape[2]}x4x2 bytes, got {len(latent_bytes)}"
			metadata = struct.pack("<IIIIII", original_width, original_height, crop_x, crop_y, latent.shape[1], latent.shape[2])

			with gzip.open(tmppath, "wb") as f:
				f.write(metadata)
				f.write(latent_bytes)
		
			tmppath.rename(encoded_path_i)


def init_worker(semaphore):
	global g_semaphore
	g_semaphore = semaphore


def encoded_path(image_hash: str, output_path: Path):
	return output_path / image_hash[:2] / image_hash[2:4] / f"{image_hash}.bin.gz"


def worker(job: tuple[str, Path]) -> dict:
	assert g_semaphore is not None
	g_semaphore.acquire()

	image_hash, image_path = job
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

	# N.B. The algorithm outlined in the SDXL paper indicates that crop_x and crop_y are expressed in the resized image's coordinate system.
	# So they represent, as they do here, the number of pixels cropped from the left and top of the resized image.
	return {
		'image': image_tensor,
		'original_width': original_size[0],
		'original_height': original_size[1],
		'crop_x': crop_x,
		'crop_y': crop_y,
		'image_hash': image_hash,
	}


if __name__ == "__main__":
	multiprocessing.set_start_method('spawn')
	main()