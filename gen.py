#!/usr/bin/env python3
"""
Generate image samples for checkpoints.
Generating them during training wastes compute and memory, so we do it on the side with other machines/GPUs.
"""
import torch
import argparse
from pathlib import Path
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler, AutoencoderKL, UNet2DConditionModel, DPMSolverMultistepScheduler
from datasets import load_dataset, DatasetDict
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as TVF
import base64
import html
from PIL import Image
import io


DEFAULT_PROMPTS = """
A photo of a dog, high quality;
Digital artwork of a dog, high quality;
disney style animation drawing of a dog;
movie poster of action superhero dog;
"""


parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="checkpoints", type=str, help="Directory with checkpoints")
parser.add_argument("--steps", default=40, type=int, help="Number of steps to run the diffusion for")
parser.add_argument("--guidance-scale", default=5.0, type=float, help="Guidance scale for the diffusion")
parser.add_argument("--prompts", default=DEFAULT_PROMPTS, type=str, help="Prompts to use for generation")
parser.add_argument("--use-ema", action="store_true", help="Use the EMA parameters")
parser.add_argument("--clip-model", type=str, default=None, help="If checkpoint doesn't have CLIP, use this model")
parser.add_argument("--refresh", action="store_true", default=False, help="Re-generate all samples")
parser.add_argument('--negative-prompt', default=None, type=str, help="Negative prompt")


base_model = "stabilityai/stable-diffusion-xl-base-1.0"
base_revision = "462165984030d82259a11f4367a4eed129e94a7b"


def main():
	args = parser.parse_args()

	if args.clip_model is None:
		args.clip_model = base_model

	# Load common models
	print("Loading tokenizers and VAE...")
	tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", revision=base_revision, use_fast=False)
	assert isinstance(tokenizer, CLIPTokenizer)
	tokenizer_2 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer_2", revision=base_revision, use_fast=False)
	assert isinstance(tokenizer_2, CLIPTokenizer)
	vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", revision="462165984030d82259a11f4367a4eed129e94a7b", torch_dtype=torch.float16, use_safetensors=True)
	assert isinstance(vae, AutoencoderKL)
	vae.eval()
	vae.to('cuda')

	# Parse prompts
	print("Parsing prompts...")
	prompts = args.prompts.split(';')
	prompts = [p.strip() for p in prompts if p.strip()]

	np.random.seed(42)

	# Find all checkpoints
	checkpoints = list(Path(args.dir).glob("**/unet"))
	checkpoints = [c.parent for c in checkpoints]
	checkpoints.sort()

	for checkpoint in tqdm(checkpoints, desc="Checkpoints"):
		dest_folder = checkpoint / ("ema_sample_images" if args.use_ema else "sample_images")
		if dest_folder.exists() and len(list(dest_folder.glob("*.png"))) >= len(prompts) and not args.refresh:
			continue

		process_checkpoint(checkpoint, prompts, vae, tokenizer, tokenizer_2, args)
	
	create_html(checkpoints, prompts, Path(args.dir))


@torch.no_grad()
def process_checkpoint(checkpoint: Path, prompts: list[str], vae: AutoencoderKL, tokenizer: CLIPTokenizer, tokenizer_2: CLIPTokenizer, args: argparse.Namespace):
	# Load checkpoint model
	print(f"Loading model from {checkpoint}...")
	if (checkpoint / "text_encoder").exists():
		text_encoder = CLIPTextModel.from_pretrained(checkpoint, subfolder="text_encoder", torch_dtype=torch.float16)
	else:
		print("Using default CLIP model for text_encoder")
		text_encoder = CLIPTextModel.from_pretrained(args.clip_model, subfolder="text_encoder", torch_dtype=torch.float16)
	assert isinstance(text_encoder, CLIPTextModel)
	if (checkpoint / "text_encoder_2").exists():
		text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(checkpoint, subfolder="text_encoder_2", torch_dtype=torch.float16)
	else:
		print("Using default CLIP model for text_encoder_2")
		text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.clip_model, subfolder="text_encoder_2", torch_dtype=torch.float16)
	assert isinstance(text_encoder_2, CLIPTextModelWithProjection)
	if args.use_ema:
		unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder="ema_unet", torch_dtype=torch.float16)
	else:
		unet = UNet2DConditionModel.from_pretrained(checkpoint, subfolder="unet", torch_dtype=torch.float16)
	assert isinstance(unet, UNet2DConditionModel)

	scheduler = EulerAncestralDiscreteScheduler.from_pretrained(base_model, subfolder="scheduler")
	scheduler = DPMSolverMultistepScheduler.from_pretrained(base_model, subfolder="scheduler", use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")

	model = StableDiffusionXLPipeline(
		vae=vae,
		text_encoder=text_encoder,
		text_encoder_2=text_encoder_2,
		tokenizer=tokenizer,
		tokenizer_2=tokenizer_2,
		unet=unet,
		add_watermarker=False,
		scheduler=scheduler,
	)

	model.unet.eval()
	model.text_encoder.eval()
	model.text_encoder_2.eval()
	model.to("cuda")

	generator = torch.Generator('cuda').manual_seed(42)
	negative_prompts = [args.negative_prompt] * len(prompts) if args.negative_prompt is not None else None
	images_pil = model(
		prompt=prompts, num_inference_steps=args.steps, guidance_scale=args.guidance_scale, generator=generator, negative_prompt=negative_prompts,
	).images

	dest_folder = checkpoint / ("ema_sample_images" if args.use_ema else "sample_images")
	dest_folder.mkdir(exist_ok=True)

	for i, image in enumerate(images_pil):
		image.save(dest_folder / f"{i}.png")


def create_html(checkpoints: list[Path], prompts: list[str], dest: Path):
	samples_folder = dest / Path("samples_html")
	samples_folder.mkdir(exist_ok=True, parents=True)

	with (samples_folder / "samples.html").open('w', encoding='utf-8') as f:
		f.write("""<html>
<head>
	<style>
		table {
			border-collapse: collapse;
			width: 100%;
		}
		tr {
			border-bottom: 1px solid #ccc;
		}
		td, th {
			text-align: center;
			padding: 5px;
		}
		img {
			width: 256px;
			height: 256px;
		}
	</style>
	<script>
		function verifyAge() {
			const dob = new Date(document.getElementById("dob").value);
			const today = new Date();
			let age = today.getFullYear() - dob.getFullYear();
			const month = today.getMonth() - dob.getMonth();
			const day = today.getDate() - dob.getDate();

			if (month < 0 || (month === 0 && day < 0)) {
				age--;
			}
		
			if (age >= 18) {
				document.getElementById("age-gate").style.display = "none";
				document.getElementById("content").style.display = "block";
			} else {
				document.getElementById("age-gate").style.display = "none";
				document.getElementById("error").style.display = "block";
			}
		}
	</script>
</head>
<body>
	<div id="age-gate">
		<h1>Age Verification</h1>
		<p>This research website contains adult content. You must be 18 years or older to view this content.</p>
		<label for="dob">Please enter your date of birth:</label><br><br>
		<input type="date" id="dob">
		<button onclick="verifyAge()">Enter</button>
	</div>
	<div id="error" style="display: none; color: red;">
		<p>Sorry, you must be 18 years or older to view this content.</p>
	</div>
	<table style="display: none;" id="content">
		<thead>
			<tr>
				<th>Checkpoint</th>""")
		for prompt in prompts:
			escaped_prompt = html.escape(prompt)
			f.write(f"<th>{escaped_prompt}</th>")
		
		f.write("</tr></thead><tbody>")

		for checkpoint in tqdm(sorted(checkpoints, key=lambda x: int(x.name.split("_")[1])), desc="Building HTML"):
			f.write("<tr>")
			escaped_checkpoint_name = html.escape(checkpoint.name)
			f.write(f"<td>{escaped_checkpoint_name}</td>")

			for i, prompt in enumerate(prompts):
				image_path = checkpoint / "sample_images" / f"{i}.png"
				with open(samples_folder / f"{checkpoint.name}_{i}.png", "wb") as fi:
					fi.write(image_path.read_bytes())
				#image = Image.open(image_path)
				#with io.BytesIO() as image_bytes:
				#	image.save(image_bytes, format="WEBP", quality=85)
				#	b64 = base64.b64encode(image_bytes.getvalue()).decode('utf-8')
				#data = image_path.read_bytes()
				#b64 = base64.b64encode(data).decode('utf-8')
				#f.write(f'<td><img src="data:image/webp;base64,{b64}"></td>')
				f.write(f'<td><img src="{checkpoint.name}_{i}.png"></td>')
			
			f.write("</tr>")
	
		f.write("</tbody></table></body></html>")


if __name__ == "__main__":
	main()