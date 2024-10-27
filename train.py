#!/usr/bin/env python3
import math
import omegaconf
import torch
import logging
from omegaconf import MISSING
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import torch.distributed
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from lamb import Lamb
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.amp.autocast_mode
import torch.backends.cuda
import torch.backends.cudnn
from transformers import get_scheduler
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
import datasets
from data import ImageDatasetPrecoded, gen_buckets, AspectBucketSampler, Bucket
from transformers.optimization import Adafactor
from torch.distributed.elastic.multiprocessing.errors import record
from contextlib import nullcontext
from diffusers.training_utils import EMAModel

import wandb

from utils import parse_args_into_config, get_cosine_schedule_with_warmup, temprngstate, distributed_rank, distributed_setup, distributed_cleanup, distributed_world_size, log_rank_0


@dataclass
class Config:
	output_dir: Path = Path("checkpoints")               # Output directory
	wandb_project: Optional[str] = None                  # Wandb project
	device_batch_size: int = 1                           # Device batch size
	batch_size: int = 2048                               # Actual batch size; gradient accumulation is used on device_batch_size to achieve this
	learning_rate: float = 1e-4                          # Learning rate
	warmup_samples: int = 100000                         # Warmup samples
	max_samples: int = 6000000                           # Max samples trained for in this session
	save_every: int = 50000                              # Save a checkpoint every n samples (approx)
	test_every: int = 50000                              # Test every n samples (approx)
	use_amp: bool = True                                 # Use automatic mixed precision
	grad_scaler: bool = True                             # Use gradient scaler
	lr_scheduler_type: str = "cosine"                    # Learning rate scheduler type
	min_lr_ratio: float = 0.0                            # Minimum learning rate ratio for scheduler
	allow_tf32: bool = True                              # Allow tf32
	seed: int = 42                                       # Random seed
	num_workers: int = 8                                 # Num workers
	stable_train_samples: int = 2048					 # Number of samples to use for stable training loss

	optimizer_type: str = "adamw"                        # Optimizer type
	adam_beta1: float = 0.9                              # Adam beta1
	adam_beta2: float = 0.999                            # Adam beta2
	adam_eps: float = 1e-8                               # Adam epsilon
	adam_weight_decay: float = 0.1                       # Adam weight decay

	clip_grad_norm: Optional[float] = 1.0                # Clip gradient norm

	dataset: str = "../data/dataset.parquet"             # Dataset path (parquet)
	vae_dir: str = "../data/vaes"                        # Directory with precomputed latents
	base_model: str = "stabilityai/stable-diffusion-xl-base-1.0"     # SDXL model to start from
	base_revision: str = "462165984030d82259a11f4367a4eed129e94a7b"  # Revision of the model
	base_variant: Optional[str] = 'fp16'				 # Variant of the model
	resume: Optional[Path] = None                        # Resume from a checkpoint
	loss_multiplier: float = 1.0                         # Loss multiplier
	train_text_encoder: bool = True                      # Train the first text encoder
	train_text_encoder_2: bool = False                   # Train the second text encoder
	offset_noise: float = 0.00						     # Offset noise (usually 0.05, disabled for now)
	ucg_rate: float = 0.1								 # UCG rate (SDXL paper specifies 0.05)
	loss_weighting: Optional[str] = None                 # None for None, 'eps' for sigma**-2, 'min-snr' for min-SNR
	gradient_checkpointing: bool = True                  # Use gradient checkpointing
	test_size: int = 2048								 # Test size
	model_dtype: str = "float32"                         # Model dtype
	use_ema: bool = False                                # Use EMA
	ema_decay: float = 0.9999                            # EMA decay
	use_ema_warmup: bool = False                         # Use EMA warmup
	ema_power: float = 2 / 3							 # EMA power
	base_text_model: Optional[str] = None                # If specified, load the text encoders from here instead of base_model
	grad_scaler_init: float = 2**16                      # Initial grad scaler
	min_snr_gamma: float = 5.0                           # Min-SNR gamma


@record
def main():
	# Logging
	logger = logging.getLogger(f'Process-{distributed_rank()}')
	logging.basicConfig(format='%(asctime)s [%(name)s] [%(levelname)s] [%(funcName)s] - %(message)s')
	logger.setLevel(logging.INFO)

	if distributed_rank() == 0:
		# Parse args
		config = parse_args_into_config(Config, logger)
		if config is None:
			torch.distributed.broadcast_object_list([None, None])
			return
		
		# Start
		wc = omegaconf.OmegaConf.to_container(config, resolve=True)
		assert isinstance(wc, dict)
		w = wandb.init(config=wc, project=config.wandb_project)
		assert w is not None
		with w:
			assert wandb.run is not None

			if wandb.run.resumed and config.resume is None:
				# Search for the folder with the highest number
				checkpoints = list(config.output_dir.glob(f"{wandb.run.id}/*"))
				checkpoints = [c.name for c in checkpoints if c.is_dir() and '_' in c.name]
				checkpoints = sorted(checkpoints, key=lambda x: int(x.split("_")[1]), reverse=True)
				if len(checkpoints) > 0:
					config.resume = config.output_dir / wandb.run.id / checkpoints[0]
					logger.info(f"WanDB run resumed, loading latest checkpoint: {config.resume}")
			
			# Broadcast the config and run_id to all other processes
			torch.distributed.broadcast_object_list([config, wandb.run.id])

			logger.info("Rank 0 starting training...")
			trainer = MainTrainer(config=config, run_id=wandb.run.id, logger=logger)
			trainer.train()
	else:
		objects = [None, None]
		logger.info(f"Rank {distributed_rank()} waiting for config...")
		torch.distributed.broadcast_object_list(objects)
		config, run_id = objects

		if config is None or run_id is None:
			logger.info(f"Rank {distributed_rank()} exiting...")
			return
		
		logger.info(f"Rank {distributed_rank()} starting training...")
		trainer = MainTrainer(config=config, run_id=run_id, logger=logger)
		trainer.train()


class MainTrainer:
	config: Config
	run_id: str
	rank: int
	logger: logging.Logger
	output_dir: Path

	train_dataset: ImageDatasetPrecoded
	stable_train_dataset: ImageDatasetPrecoded
	validation_dataset: ImageDatasetPrecoded | None
	train_buckets: list[Bucket]
	stable_buckets: list[Bucket]
	validation_buckets: list[Bucket]

	train_sampler: AspectBucketSampler
	stable_train_sampler: AspectBucketSampler
	validation_sampler: AspectBucketSampler
	train_dataloader: DataLoader
	stable_train_dataloader: DataLoader
	validation_dataloader: DataLoader | None

	optimizer: Optimizer
	device: str
	device_batch_size: int
	gradient_accumulation_steps: int
	test_every_step: int
	save_every_step: int
	total_steps: int
	total_device_batches: int
	unet: UNet2DConditionModel | torch.nn.parallel.DistributedDataParallel
	text_encoder: CLIPTextModel | torch.nn.parallel.DistributedDataParallel
	text_encoder_2: CLIPTextModelWithProjection | torch.nn.parallel.DistributedDataParallel
	tokenizer: CLIPTokenizer
	tokenizer_2: CLIPTokenizer

	def __init__(self, config: Config, run_id: str, logger: logging.Logger):
		dtypes = { "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16 }
		self.config = config
		self.rank = distributed_rank()
		self.run_id = run_id
		self.logger = logger
		self.output_dir = Path(config.output_dir)
		self.device = f"cuda:{torch.cuda.current_device()}"
		self.world_size = distributed_world_size()
		self.global_dtype = dtypes[config.model_dtype]

		if config.allow_tf32:
			torch.backends.cuda.matmul.allow_tf32 = True
			torch.backends.cudnn.allow_tf32 = True
		
		self.device_batch_size = min(config.batch_size // self.world_size, config.device_batch_size)
		self.gradient_accumulation_steps = config.batch_size // (self.device_batch_size * self.world_size)
		self.test_every_step = int(math.ceil(config.test_every / config.batch_size))
		self.save_every_step = int(math.ceil(config.save_every / config.batch_size))
		self.total_steps = self.config.max_samples // self.config.batch_size
		self.total_device_batches = self.total_steps * self.gradient_accumulation_steps

		assert config.batch_size == self.device_batch_size * self.gradient_accumulation_steps * self.world_size, "Batch size must be a multiple of device batch size"
	
	def build_dataset(self):
		log_rank_0(self.logger, logging.INFO, "Building dataset...")

		source_ds = datasets.load_dataset("parquet", data_files=self.config.dataset)
		assert isinstance(source_ds, datasets.DatasetDict)

		# DEBUG: TEMP: REMOVE
		#source_ds = source_ds.filter(lambda x: (Path(self.config.vae_dir) / x['image_hash'][:2] / x['image_hash'][2:4] / f"{x['image_hash']}.bin.gz").exists())
		#log_rank_0(self.logger, logging.INFO, f"Filtered dataset to {len(source_ds['train'])} samples")

		# Split the dataset into train and test
		source_ds = source_ds['train'].train_test_split(test_size=self.config.test_size, seed=42)

		stable_train_dataset = source_ds["train"].shuffle(hash((self.config.seed, 'stable_train')) & 0xffffffff).select(range(self.config.stable_train_samples))

		log_rank_0(self.logger, logging.INFO, f"Train size: {len(source_ds['train'])}, Test size: {len(source_ds['test'])}")

		self.train_dataset = ImageDatasetPrecoded(source_ds['train'], self.tokenizer, self.tokenizer_2, datapath=self.config.vae_dir)
		self.stable_train_dataset = ImageDatasetPrecoded(stable_train_dataset, self.tokenizer, self.tokenizer_2, datapath=self.config.vae_dir)
		self.validation_dataset = ImageDatasetPrecoded(source_ds['test'], self.tokenizer, self.tokenizer_2, datapath=self.config.vae_dir)

		log_rank_0(self.logger, logging.INFO, "Building aspect ratio buckets...")
		self.train_buckets = gen_buckets(source_ds['train'])
		self.validation_buckets = gen_buckets(source_ds['test'])
		self.stable_buckets = gen_buckets(stable_train_dataset)

		if self.rank == 0:
			log_rank_0(self.logger, logging.INFO, "Aspect ratio buckets:")
			for bucket in self.train_buckets:
				log_rank_0(self.logger, logging.INFO, f"{bucket.resolution}: {len(bucket.images)}")
			log_rank_0(self.logger, logging.INFO, "")
	
	def build_dataloader(self):
		log_rank_0(self.logger, logging.INFO, "Building dataloader...")

		self.train_sampler = AspectBucketSampler(dataset=self.train_dataset, buckets=self.train_buckets, batch_size=self.device_batch_size, num_replicas=self.world_size, rank=self.rank, shuffle=True, ragged_batches=False)
		self.stable_train_sampler = AspectBucketSampler(dataset=self.stable_train_dataset, buckets=self.stable_buckets, batch_size=self.device_batch_size, num_replicas=self.world_size, rank=self.rank, shuffle=False, ragged_batches=True)
		self.validation_sampler = AspectBucketSampler(dataset=self.validation_dataset, buckets=self.validation_buckets, batch_size=self.device_batch_size, num_replicas=self.world_size, rank=self.rank, shuffle=False, ragged_batches=True)

		self.train_dataloader = DataLoader(
			self.train_dataset,
			batch_sampler=self.train_sampler,
			num_workers=self.config.num_workers,
			collate_fn=self.train_dataset.collate_fn,
			pin_memory=True,
			pin_memory_device=self.device,
		)

		self.stable_train_dataloader = DataLoader(
			self.stable_train_dataset,
			batch_sampler=self.stable_train_sampler,
			num_workers=self.config.num_workers,
			collate_fn=self.stable_train_dataset.collate_fn,
			pin_memory=True,
			pin_memory_device=self.device,
		)

		if self.validation_dataset is not None:
			self.validation_dataloader = DataLoader(
				self.validation_dataset,
				batch_sampler=self.validation_sampler,
				num_workers=self.config.num_workers,
				collate_fn=self.validation_dataset.collate_fn,
				pin_memory=True,
				pin_memory_device=self.device,
			)
		else:
			self.validation_dataloader = None
	
	def build_model(self):
		log_rank_0(self.logger, logging.INFO, "Building model...")
		base_model = self.config.base_model
		base_revision = self.config.base_revision
		variant = self.config.base_variant
		resume_model = base_model
		resume_revision = base_revision

		if self.config.resume is not None:
			resume_model = self.config.resume
			resume_revision = "main"
			variant = None

		log_rank_0(self.logger, logging.INFO, "Loading tokenizer...")
		self.tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer", revision=base_revision, use_fast=False)
		self.tokenizer_2 = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer_2", revision=base_revision, use_fast=False)

		log_rank_0(self.logger, logging.INFO, "Loading text encoder...")
		if self.config.resume is not None and (Path(resume_model) / "text_encoder").exists():
			log_rank_0(self.logger, logging.INFO, f"Loading text encoder from {resume_model}")
			text_encoder = CLIPTextModel.from_pretrained(resume_model, subfolder="text_encoder", revision=resume_revision, torch_dtype=self.global_dtype, variant=variant, use_safetensors=True)
		elif self.config.base_text_model is not None:
			log_rank_0(self.logger, logging.INFO, f"Loading text encoder from {self.config.base_text_model}")
			text_encoder = CLIPTextModel.from_pretrained(self.config.base_text_model, subfolder="text_encoder", revision="main", torch_dtype=self.global_dtype)
		else:
			log_rank_0(self.logger, logging.INFO, f"Loading text encoder from {base_model}")
			text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder", revision=base_revision, torch_dtype=self.global_dtype, variant=variant, use_safetensors=True)
		assert isinstance(text_encoder, CLIPTextModel)
		self.text_encoder = text_encoder

		if self.config.resume is not None and (Path(resume_model) / "text_encoder_2").exists():
			log_rank_0(self.logger, logging.INFO, f"Loading text encoder 2 from {resume_model}")
			text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(resume_model, subfolder="text_encoder_2", revision=resume_revision, torch_dtype=self.global_dtype, variant=variant, use_safetensors=True)
		elif self.config.base_text_model is not None:
			log_rank_0(self.logger, logging.INFO, f"Loading text encoder 2 from {self.config.base_text_model}")
			text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(self.config.base_text_model, subfolder="text_encoder_2", revision="main", torch_dtype=self.global_dtype)
		else:
			log_rank_0(self.logger, logging.INFO, f"Loading text encoder 2 from {base_model}")
			text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(base_model, subfolder="text_encoder_2", revision=base_revision, torch_dtype=self.global_dtype, variant=variant, use_safetensors=True)
		assert isinstance(text_encoder_2, CLIPTextModelWithProjection)
		self.text_encoder_2 = text_encoder_2

		self.logger.info("Loading UNet...")
		unet = UNet2DConditionModel.from_pretrained(resume_model, subfolder="unet", revision=resume_revision, torch_dtype=self.global_dtype, variant=variant, use_safetensors=True)
		assert isinstance(unet, UNet2DConditionModel)
		self.unet = unet

		# EMA
		# All ranks need a copy for validation to work
		if self.config.use_ema:
			self.ema_unet = EMAModel(self.unet.parameters(), model_cls=UNet2DConditionModel, model_config=self.unet.config, decay=self.config.ema_decay, power=self.config.ema_power, use_ema_warmup=self.config.use_ema_warmup)
			if self.config.resume is not None:
				load_model = EMAModel.from_pretrained(Path(self.config.resume) / "ema_unet", UNet2DConditionModel)
				self.ema_unet.load_state_dict(load_model.state_dict())
				del load_model

			if self.config.train_text_encoder:
				self.ema_text_encoder = EMAModel(self.text_encoder.parameters(), model_cls=CLIPTextModel, model_config=self.text_encoder.config, decay=self.config.ema_decay, power=self.config.ema_power, use_ema_warmup=self.config.use_ema_warmup)
			
				if self.config.resume is not None:
					load_model = EMAModel.from_pretrained(Path(self.config.resume) / "ema_text_encoder", CLIPTextModel)
					self.ema_text_encoder.load_state_dict(load_model.state_dict())
					del load_model
			
			if self.config.train_text_encoder_2:
				self.ema_text_encoder_2 = EMAModel(self.text_encoder_2.parameters(), model_cls=CLIPTextModelWithProjection, model_config=self.text_encoder_2.config, decay=self.config.ema_decay, power=self.config.ema_power, use_ema_warmup=self.config.use_ema_warmup)

				if self.config.resume is not None:
					load_model = EMAModel.from_pretrained(Path(self.config.resume) / "ema_text_encoder_2", CLIPTextModelWithProjection)
					self.ema_text_encoder_2.load_state_dict(load_model.state_dict())
					del load_model

		if self.config.gradient_checkpointing:
			self.unet.enable_gradient_checkpointing()

		log_rank_0(self.logger, logging.INFO, "Loading noise scheduler...")
		#self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False)
		self.noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler", revision=base_revision)
		sigmas = ((1 - self.noise_scheduler.alphas_cumprod) / self.noise_scheduler.alphas_cumprod)**0.5
		self.sigmas = torch.flip(sigmas, (0,))
		self.sigmas = self.sigmas.to(self.device, dtype=self.global_dtype)
		self.sigmas.requires_grad_(False)

		# Min-SNR
		alphas_cumprod = self.noise_scheduler.alphas_cumprod
		all_snr = alphas_cumprod / (1 - alphas_cumprod)
		all_snr.requires_grad_(False)
		self.all_snr = all_snr.to(self.device, dtype=torch.float32)

		self.text_encoder.requires_grad_(self.config.train_text_encoder)
		self.text_encoder_2.requires_grad_(self.config.train_text_encoder_2)
		self.unet.requires_grad_(True)

		log_rank_0(self.logger, logging.INFO, "Moving models to device...")
		self.text_encoder.to(self.device, dtype=self.global_dtype) # type: ignore
		self.text_encoder_2.to(self.device, dtype=self.global_dtype) # type: ignore
		self.unet.to(self.device, dtype=self.global_dtype)

		if self.config.use_ema:
			self.ema_unet.to(self.device)

			if self.config.train_text_encoder:
				self.ema_text_encoder.to(self.device)
			
			if self.config.train_text_encoder_2:
				self.ema_text_encoder_2.to(self.device)

		# Distributed training
		if self.world_size > 1:
			self.unet = torch.nn.parallel.DistributedDataParallel(self.unet, device_ids=[self.rank], output_device=self.rank, gradient_as_bucket_view=True, find_unused_parameters=True)
			if self.config.train_text_encoder:
				self.text_encoder = torch.nn.parallel.DistributedDataParallel(self.text_encoder, device_ids=[self.rank], output_device=self.rank, gradient_as_bucket_view=True, find_unused_parameters=True)
			if self.config.train_text_encoder_2:
				self.text_encoder_2 = torch.nn.parallel.DistributedDataParallel(self.text_encoder_2, device_ids=[self.rank], output_device=self.rank, gradient_as_bucket_view=True, find_unused_parameters=True)
			log_rank_0(self.logger, logging.INFO, "DistributedDataParallel wrapped")

	def build_optimizer(self):
		log_rank_0(self.logger, logging.INFO, "Building optimizer...")
		self.optimized_params = list(self.unet.parameters())
		if self.config.train_text_encoder:
			self.optimized_params += list(self.text_encoder.parameters())
		if self.config.train_text_encoder_2:
			self.optimized_params += list(self.text_encoder_2.parameters())

		if self.config.optimizer_type == "adamw":
			optimizer_cls = torch.optim.AdamW
			kwargs = {
				'lr': self.config.learning_rate,
				'betas': (self.config.adam_beta1, self.config.adam_beta2),
				'eps': self.config.adam_eps,
				'weight_decay': self.config.adam_weight_decay,
			}
		elif self.config.optimizer_type == 'lamb':
			optimizer_cls = Lamb
			kwargs = {
				'lr': self.config.learning_rate,
				'betas': (self.config.adam_beta1, self.config.adam_beta2),
				'eps': self.config.adam_eps,
				'weight_decay': self.config.adam_weight_decay,
			}
		elif self.config.optimizer_type == 'fusedlamb':
			from apex.optimizers import FusedLAMB # type: ignore
			optimizer_cls = FusedLAMB
			kwargs = {
				'lr': self.config.learning_rate,
				'betas': (self.config.adam_beta1, self.config.adam_beta2),
				'eps': self.config.adam_eps,
				'weight_decay': self.config.adam_weight_decay,
			}
		elif self.config.optimizer_type == 'adafactor':
			optimizer_cls = Adafactor
			kwargs = {
				'lr': self.config.learning_rate,
				'weight_decay': self.config.adam_weight_decay,
				'relative_step': False,
				'scale_parameter': False,
				'warmup_init': False,
			}
		elif self.config.optimizer_type == 'came':
			from came_pytorch import CAME
			optimizer_cls = CAME
			kwargs = {
				'lr': self.config.learning_rate,
				'betas': (0.9, 0.999, 0.9999),
				'eps': (1e-30, 1e-16),
				'weight_decay': self.config.adam_weight_decay,
			}
		else:
			raise ValueError(f"Unknown optimizer type {self.config.optimizer_type}")
		
		self.optimizer = optimizer_cls(self.optimized_params, **kwargs)
	
	def build_lr_scheduler(self):
		self.logger.info("Building lr scheduler...")
		num_warmup_steps = int(math.ceil(self.config.warmup_samples / self.config.batch_size))

		if self.config.lr_scheduler_type == "cosine":
			self.lr_scheduler = get_cosine_schedule_with_warmup(
				optimizer=self.optimizer,
				num_warmup_steps=num_warmup_steps,
				num_training_steps=self.total_steps,
				min_lr_ratio=self.config.min_lr_ratio,
			)
		else:
			self.lr_scheduler = get_scheduler(self.config.lr_scheduler_type, self.optimizer, num_warmup_steps, self.total_steps)
		#else:
		#	raise ValueError(f"Unknown lr scheduler type {self.config.lr_scheduler_type}")
	
	def train(self):
		# Seed
		seed = hash((self.config.seed, self.rank)) & 0xffffffff   # NumPy requires 32-bit seeds
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)

		#self.scaler = CustomGradScaler(enabled=self.config.grad_scaler, init_scale=self.config.grad_scaler_init)
		self.scaler = torch.amp.grad_scaler.GradScaler(self.device, enabled=self.config.grad_scaler, init_scale=self.config.grad_scaler_init)
		self.build_model()
		self.build_dataset()
		self.build_dataloader()
		self.build_optimizer()
		self.build_lr_scheduler()

		device_step = 0

		# Resume
		if self.config.resume is not None:
			resume = torch.load(Path(self.config.resume) / "training_state.pt", map_location='cpu')
			resume.update(torch.load(Path(self.config.resume) / f"training_state{self.rank}.pt", map_location='cpu'))  # Load rank-specific state

			self.lr_scheduler.load_state_dict(resume["lr_scheduler"])

			random.setstate(resume["random_state"])
			np.random.set_state(resume["np_random_state"])
			torch.random.set_rng_state(resume["torch_random_state"])
			try:
				torch.cuda.random.set_rng_state(resume["torch_cuda_random_state"])
			except RuntimeError:
				self.logger.warning("Failed to restore cuda random state, this is normal if you're using a different number of GPUs than last time")

			self.optimizer.load_state_dict(resume["optimizer"])

			#resume['scaler']['_growth_tracker'] = 240

			try:
				self.scaler.load_state_dict(resume["scaler"])
			except RuntimeError:
				self.logger.warning("Failed to restore scaler state, possibly old bugged save")

			device_step = (resume["global_step"] + 1) * self.gradient_accumulation_steps

			self.train_sampler.set_state(resume["sampler_epoch"], resume["sampler_index"])

			del resume
		
		self.scaler.set_growth_interval(500000 // self.config.batch_size)

		# Compile model
		#self.text_encoder = torch.compile(self.text_encoder)
		#self.text_encoder_2 = torch.compile(self.text_encoder_2)
		#self.unet = torch.compile(self.unet, mode="reduce-overhead", fullgraph=True)
		#self.compiled_text_encoder = self.text_encoder
		#self.compiled_unet = self.unet

		# Wandb
		#if self.rank == 0:
			#wandb.watch(self.embedder, log_freq=100)
		
		# Initial validation and saving, for debugging
		if device_step == 0:
			self.global_step = device_step // self.gradient_accumulation_steps
			self.global_samples_seen = device_step * self.device_batch_size * self.world_size
			self.save_checkpoint()
			self.do_validation()

		self.logger.info("Training...")
		loss_sum = torch.tensor(0.0, device='cuda', requires_grad=False, dtype=torch.float32)
		dataloader_iter = iter(self.train_dataloader)

		pbar = tqdm(total=self.total_device_batches * self.device_batch_size * self.world_size, initial=device_step * self.device_batch_size * self.world_size, dynamic_ncols=True, smoothing=0.01, disable=self.rank != 0)
		with logging_redirect_tqdm():
			for device_step in range(device_step, self.total_device_batches):
				self.global_step = device_step // self.gradient_accumulation_steps
				self.global_samples_seen = (device_step + 1) * self.device_batch_size * self.world_size

				self.unet.train()
				if self.config.train_text_encoder:
					self.text_encoder.train()
				else:
					self.text_encoder.eval()
				if self.config.train_text_encoder_2:
					self.text_encoder_2.train()
				else:
					self.text_encoder_2.eval()

				# Get batch
				try:
					batch = next(dataloader_iter)
				except StopIteration:
					logging.warning("Dataloader iterator exhausted, restarting...")
					self.train_sampler.set_epoch(self.train_sampler.epoch + 1)
					dataloader_iter = iter(self.train_dataloader)
					batch = next(dataloader_iter)

				# Move batch to device
				batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

				is_last_device_step = (device_step + 1) % self.gradient_accumulation_steps == 0
				is_last_step = (self.global_step + 1) == self.total_steps

				# Forward pass
				# Disable gradient sync for all but the last device step
				no_sync_gradients = not is_last_device_step and self.world_size > 1
				with self.unet.no_sync() if no_sync_gradients else nullcontext(), self.text_encoder.no_sync() if no_sync_gradients and self.config.train_text_encoder else nullcontext(), self.text_encoder_2.no_sync() if no_sync_gradients and self.config.train_text_encoder_2 else nullcontext():
					loss = self.get_model_pred(batch, cfg_dropping=True, disable_loss_weighting=False)
				
					loss = loss.float() / self.gradient_accumulation_steps
					loss_sum.add_(loss.detach())

					if torch.isnan(loss) or torch.isinf(loss):
						self.logger.error("ERROR: Loss is NaN or Inf")
						exit()
					
					# Backward pass
					self.scaler.scale(loss).backward() # type: ignore

				# Take a step if accumulation is done
				if is_last_device_step:
					log_rank_0(self.logger, logging.INFO, f"Step {self.global_step + 1}, loss: {loss_sum.item()}")

					# Reduce loss_sum across devices for logging
					torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)

					# Unscale the gradients before clipping
					self.scaler.unscale_(self.optimizer)

					# Clip gradients
					if self.config.clip_grad_norm is not None:
						torch.nn.utils.clip_grad.clip_grad_norm_(self.optimized_params, self.config.clip_grad_norm)

					# Take a step
					self.scaler.step(self.optimizer)
					self.scaler.update()
					self.lr_scheduler.step()
					self.optimizer.zero_grad(set_to_none=True)
					#self.optimizer.zero_grad()

					if self.config.use_ema:
						self.ema_unet.step(self.unet.parameters())
						if self.config.train_text_encoder:
							self.ema_text_encoder.step(self.text_encoder.parameters())
						if self.config.train_text_encoder_2:
							self.ema_text_encoder_2.step(self.text_encoder_2.parameters())

					if self.rank == 0:
						logs = {
							"train/loss": loss_sum.item() / (self.config.loss_multiplier * self.world_size),
							"train/lr": self.lr_scheduler.get_last_lr()[0],
							"train/samples": self.global_samples_seen,
							"train/scaler": self.scaler.get_scale(),
						}
						wandb.log(logs, step=self.global_step)
					
					loss_sum.zero_()
				
					# Save checkpoint
					# Saved every save_every steps and at the end of training
					if self.save_every_step > 0 and ((self.global_step + 1) % self.save_every_step == 0 or is_last_step):
						self.save_checkpoint()

					# Validation
					# Run every test_every steps and at the end of training
					if self.test_every_step > 0 and ((self.global_step + 1) % self.test_every_step == 0 or is_last_step):
						self.do_validation()
						self.do_stable_train_loss()
				
				pbar.update(self.device_batch_size * self.world_size)
			
			pbar.close()
	
	def get_model_pred(self, batch, cfg_dropping: bool, disable_loss_weighting: bool) -> torch.Tensor:
		with torch.amp.autocast_mode.autocast('cuda', enabled=self.config.use_amp):
			latents = batch['latent'].to(self.global_dtype)

			# Noise
			noise = torch.randn_like(latents)
			noise = noise + self.config.offset_noise * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device, dtype=latents.dtype)
			bsz = latents.shape[0]
			timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device) # type: ignore
			timesteps = timesteps.long()

			noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps) # type: ignore

			# Encode the prompt(s)
			# batch['prompt'] is expected to be (BxNx77)
			# Squash the first two dimensions to process everything in parallel
			assert batch["prompt"].shape[2] == 77 and batch["prompt"].shape[0] == bsz
			n = batch['prompt'].shape[1]
			prompt = batch["prompt"].view(-1, 77)
			prompt_2 = batch["prompt_2"].view(-1, 77)

			#self.logger.info(f"DEBUG: prompt: {prompt.shape}")

			prompt_embed1 = self.text_encoder(prompt, output_hidden_states=True).hidden_states[-2]
			prompt_embed2 = self.text_encoder_2(prompt_2, output_hidden_states=True)
			pooled_prompt_embeds = prompt_embed2[0]
			prompt_embed2 = prompt_embed2.hidden_states[-2]

			# Unsquash to BxN*77
			prompt_embed1 = prompt_embed1.view(bsz, -1, prompt_embed1.shape[-1])
			prompt_embed2 = prompt_embed2.view(bsz, -1, prompt_embed2.shape[-1])
			pooled_prompt_embeds = pooled_prompt_embeds.view(bsz, -1, pooled_prompt_embeds.shape[-1])
			assert prompt_embed1.shape == (bsz, n*77, 768) and prompt_embed2.shape == (bsz, n*77, 1280)
			assert pooled_prompt_embeds.shape == (bsz, n, 1280)

			# Concat the two embeddings along the last dimension (BxN*77x(768+1280))
			prompt_embed = torch.concat([prompt_embed1, prompt_embed2], dim=-1)

			# Only the first pooled_prompt_embeds
			# That seems to be how ComfyUI inference does it, but I wonder if there is something better? Average?
			pooled_prompt_embeds = pooled_prompt_embeds[:, 0, :]
			assert pooled_prompt_embeds.shape == (bsz, 1280)

			#if cfg_dropping:
			#	mask = torch.rand(bsz, device=self.device) > (self.config.ucg_rate / 2)
			#	prompt_embed = prompt_embed * mask[:, None, None]
			#	pooled_prompt_embeds = pooled_prompt_embeds * mask[:, None]

			# SDXL (according to HF) seems to use zero as the negative embedding (when there's no negative prompt)
			# This seems odd to me, since most end users will instead use an empty prompt for the negative embedding
			# Don't we want empty prompt to mean unconditioned, not "images with no prompt in the training set"?
			# To split the difference, we'll randomly choose the dropping method
			# if cfg_dropping:
			# 	# Embed ""
			# 	empty_prompt = self.empty_prompt.clone().detach().unsqueeze(0).repeat(bsz*n, 1)
			# 	empty_prompt_2 = self.empty_prompt_2.clone().detach().unsqueeze(0).repeat(bsz*n, 1)

			# 	empty_embed1 = self.text_encoder(empty_prompt, output_hidden_states=True).hidden_states[-2]
			# 	empty_embed1 = 
			# 	assert empty_embed1.shape == (bsz, 77, 768)
			# 	empty_embed2 = self.text_encoder_2(empty_prompt_2, output_hidden_states=True)
			# 	pooled_empty_embeds = empty_embed2[0]
			# 	assert pooled_empty_embeds.shape == (bsz, 1280)
			# 	empty_embed2 = empty_embed2.hidden_states[-2]
			# 	assert empty_embed2.shape == (bsz, 77, 1280)
			# 	empty_embeds = torch.concatenate([empty_embed1, empty_embed2], dim=-1)

			# 	# Mask determines which embeddings will be dropped, while drop_type determines if they'll be zeroed or replaced with the empty prompt
			# 	mask = torch.rand(bsz, device=prompt_embed.device) > self.config.ucg_rate
			# 	drop_type = torch.rand(bsz, device=prompt_embed.device) < 0.5

			# 	# Expand to match prompt_embed (BxN*77x...)
			# 	# ComfyUI seems to use repeat here
			# 	empty_embeds = empty_embeds.repeat(1, n, 1)
			# 	assert empty_embeds.shape == prompt_embed.shape, f"{empty_embeds.shape} != {prompt_embed.shape}"

			# 	# Drop the embeddings
			# 	prompt_embed = prompt_embed * mask[:, None, None] + empty_embeds * (~mask)[:, None, None] * drop_type[:, None, None]
			# 	pooled_prompt_embeds = pooled_prompt_embeds * mask[:, None] + pooled_empty_embeds * (~mask[:, None]) * drop_type[:, None]

			add_text_embeds = pooled_prompt_embeds

			# SDXL Micro-conditioning: original size, crop, and target size
			# SAI's code seems to apply UCG dropping to this conditioning
			# But it doesn't look like inference code ever uses this. The negative micro-conditioning is always the same as the positive.
			# Also, it's difficult because the embeddings themselves need to be dropped, not the sizes.
			# And Kohya scripts don't do it.  So, meh.
			assert batch['original_size'].shape == (bsz, 2)
			add_time_ids = torch.cat((batch['original_size'], batch['crop'], batch['target_size']), 1)
			assert add_time_ids.shape == (bsz, 6)

			if not hasattr(self, 'debug_write_training_batch') and self.rank == 0:
				self.debug_write_training_batch = True
				torch.save({
					"noisy_latents": noisy_latents,
					"timesteps": timesteps,
					"prompt_embed": prompt_embed,
					"add_text_embeds": add_text_embeds,
					"add_time_ids": add_time_ids,
					"noise": noise,
					"batch": batch,
					#"mask": mask if cfg_dropping else None,
					#"drop_type": drop_type if cfg_dropping else None,
					#"empty_embeds": empty_embeds if cfg_dropping else None,
					"pooled_prompt_embeds": pooled_prompt_embeds,
				}, "debug_training_batch.pt")
			
			#self.logger.info(f"Training shapes: {noisy_latents.shape}, {timesteps.shape}, {prompt_embed.shape}, {add_text_embeds.shape}, {add_time_ids.shape}")

			# Forward pass
			model_pred = self.unet(
				noisy_latents,
				timesteps,
				encoder_hidden_states=prompt_embed,
				added_cond_kwargs={
					"text_embeds": add_text_embeds,
					"time_ids": add_time_ids,
				},
				return_dict=False,
			)[0]
			
			# Loss (epsilon prediction)
			loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
			loss = loss.mean(dim=(1, 2, 3))
			assert loss.shape == (bsz,)

			# This was in the SDXL github repo. Not sure what it does exactly; I couldn't find any official mention of it anywhere
			# but it's used in the default training example.
			# That said, the SDXL training codebase is ... odd.
			# NOTE: Now with min-SNR implemented, it looks like the SDXL github repo is doing SNR weight, but without the min gamma thing.
			if not disable_loss_weighting:
				if self.config.loss_weighting == 'eps':
					loss = loss * (self.sigmas[timesteps].float() ** -2)
				elif self.config.loss_weighting == 'min-snr':
					snr = self.all_snr[timesteps]
					min_snr_gamma = torch.minimum(snr, torch.full_like(snr, self.config.min_snr_gamma))
					snr_weight = min_snr_gamma / snr

					loss = loss * snr_weight
				elif self.config.loss_weighting is not None:
					raise ValueError(f"Unknown loss weighting type {self.config.loss_weighting}")
			
			loss = loss.mean()
			assert loss.shape == ()

			return loss * self.config.loss_multiplier
	
	def save_checkpoint(self):
		log_rank_0(self.logger, logging.INFO, "Saving checkpoint...")

		sampler_epoch = self.train_sampler.epoch
		sampler_index = self.global_samples_seen // self.world_size  # NOTE: sampler_index is in terms of "samples", not batches or steps
		sampler_index = sampler_index % (len(self.train_dataloader) * self.device_batch_size)

		base_path = self.output_dir / self.run_id
		path = base_path / f"samples_{self.global_samples_seen}"
		tmp_path = base_path / "tmp"

		tmp_path.mkdir(parents=True, exist_ok=True)

		if self.rank == 0:
			unet = self.unet.module if isinstance(self.unet, torch.nn.parallel.DistributedDataParallel) else self.unet
			text_encoder = self.text_encoder.module if isinstance(self.text_encoder, torch.nn.parallel.DistributedDataParallel) else self.text_encoder
			text_encoder_2 = self.text_encoder_2.module if isinstance(self.text_encoder_2, torch.nn.parallel.DistributedDataParallel) else self.text_encoder_2

			log_rank_0(self.logger, logging.INFO, f"Saving checkpoint to {path}...")
			unet.save_pretrained(tmp_path / "unet", safe_serialization=True)
			if self.config.train_text_encoder:
				text_encoder.save_pretrained(tmp_path / "text_encoder", safe_serialization=True)
			if self.config.train_text_encoder_2:
				text_encoder_2.save_pretrained(tmp_path / "text_encoder_2", safe_serialization=True)
			
			if self.config.use_ema:
				self.ema_unet.save_pretrained(tmp_path / "ema_unet")
				if self.config.train_text_encoder:
					self.ema_text_encoder.save_pretrained(tmp_path / "ema_text_encoder")
				if self.config.train_text_encoder_2:
					self.ema_text_encoder_2.save_pretrained(tmp_path / "ema_text_encoder_2")

			torch.save({
				"global_step": self.global_step,
				"global_samples_seen": self.global_samples_seen,
				"optimizer": self.optimizer.state_dict(),
				"lr_scheduler": self.lr_scheduler.state_dict(),
				"sampler_epoch": sampler_epoch,
				"sampler_index": sampler_index,
			}, tmp_path / "training_state.pt")
		
		# Rank dependent stuff
		torch.save({
			"global_step": self.global_step,
			"global_samples_seen": self.global_samples_seen,
			"scaler": self.scaler.state_dict(),
			"random_state": random.getstate(),
			"np_random_state": np.random.get_state(),
			"torch_random_state": torch.random.get_rng_state(),
			"torch_cuda_random_state": torch.cuda.random.get_rng_state(),
		}, tmp_path / f"training_state{self.rank}.pt")

		# Sync all processes before moving files
		if self.world_size > 1:
			torch.distributed.barrier()
		
		# Move checkpoint into place
		if self.rank == 0:
			tmp_path.rename(path)
	
	@torch.no_grad()
	def do_validation(self):
		if self.validation_dataloader is None:
			return
		
		# Perform validation using the ema version, storing the original parameters
		if self.config.use_ema:
			self.ema_unet.store(self.unet.parameters())
			self.ema_unet.copy_to(self.unet.parameters())
			if self.config.train_text_encoder:
				self.ema_text_encoder.store(self.text_encoder.parameters())
				self.ema_text_encoder.copy_to(self.text_encoder.parameters())
			if self.config.train_text_encoder_2:
				self.ema_text_encoder_2.store(self.text_encoder_2.parameters())
				self.ema_text_encoder_2.copy_to(self.text_encoder_2.parameters())

		log_rank_0(self.logger, logging.INFO, "Running validation...")

		self.unet.eval()
		self.text_encoder.eval()
		self.text_encoder_2.eval()

		total_loss = torch.tensor(0.0, device=self.device, requires_grad=False, dtype=torch.float32)

		# Set seed for reproducibility
		with temprngstate(42), tqdm(total=len(self.validation_dataloader) * self.device_batch_size, dynamic_ncols=True, desc="Validation", disable=self.rank != 0) as pbar:
			for batch in self.validation_dataloader:
				# Move batch to device
				batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
				
				# Forward pass
				loss = self.get_model_pred(batch, cfg_dropping=False, disable_loss_weighting=True)
				
				total_loss.add_(loss.detach())
				pbar.update(self.device_batch_size)
		
		total_loss = total_loss / len(self.validation_dataloader)
		torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
		total_loss = total_loss / self.world_size
		total_loss = total_loss / self.config.loss_multiplier

		# Switch back to original parameters
		if self.config.use_ema:
			self.ema_unet.restore(self.unet.parameters())
			if self.config.train_text_encoder:
				self.ema_text_encoder.restore(self.text_encoder.parameters())
			if self.config.train_text_encoder_2:
				self.ema_text_encoder_2.restore(self.text_encoder_2.parameters())

		# All other ranks are done
		if self.rank != 0:
			return

		wandb.log({
			"validation/samples": self.global_samples_seen,
			"validation/scaler": self.scaler.get_scale(),
			"validation/loss": total_loss.item(),
		}, step=self.global_step)
	
	@torch.no_grad()
	def do_stable_train_loss(self):
		"""
		Calculates a stable version of the training loss by using a fixed set of training samples.
		Useful for tracking training loss when the normal training loss is noisy.
		"""
		log_rank_0(self.logger, logging.INFO, "Running stable train loss...")

		self.unet.eval()
		self.text_encoder.eval()
		self.text_encoder_2.eval()

		total_loss = torch.tensor(0.0, device=self.device, requires_grad=False, dtype=torch.float32)

		# Set seed for reproducibility
		with temprngstate(42), tqdm(total=len(self.stable_train_dataloader) * self.device_batch_size, dynamic_ncols=True, desc="Stable Train Loss", disable=self.rank != 0) as pbar:
			for batch in self.stable_train_dataloader:
				# Move batch to device
				batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
				
				# Forward pass
				loss = self.get_model_pred(batch, cfg_dropping=False, disable_loss_weighting=False)
				
				total_loss.add_(loss.detach())
				pbar.update(self.device_batch_size)
		
		total_loss = total_loss / len(self.stable_train_dataloader)
		torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
		total_loss = total_loss / self.world_size
		total_loss = total_loss / self.config.loss_multiplier

		if self.rank != 0:
			return
		
		wandb.log({
			"stable_train_loss/samples": self.global_samples_seen,
			"stable_train_loss/scaler": self.scaler.get_scale(),
			"stable_train_loss/loss": total_loss.item(),
		}, step=self.global_step)


class CustomGradScaler(torch.cuda.amp.grad_scaler.GradScaler):
	"""
	GradScaler that forces allow_fp16 to be True
	I'm not entirely sure why pytorch has this parameter set to False?
	But we need it, since our gradients are in fp16
	"""
	def _unscale_grads_(self, optimizer, inv_scale, found_inf, allow_fp16):
		# Force allow_fp16 to be True
		return super()._unscale_grads_(optimizer, inv_scale, found_inf, True)


if __name__ == '__main__':
	distributed_setup()
	torch.cuda.set_device(distributed_rank())
	main()
	distributed_cleanup()