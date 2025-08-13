#!/usr/bin/env python
"""
# Manual Image Head to Head Scoring
This builds a dataset that we can then hopefully use ELO on to eventually score images.
"""
import asyncio
import base64
import itertools
import json
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import asyncpg
import magic
import openai
import torch
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from grid_models import NsfwClassifier as GridClassifier
from models import QualityClassifier
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import trueskill as ts
from math import log2
import numpy as np


DATASET_LEN = 4000

SYSTEM_PROMPT = """
...
"""

USER_PROMPT = """..."""


class Settings(BaseSettings):
	pg_socket: Path = Path.cwd().parent.parent / "pg-socket"

	model_config = SettingsConfigDict(env_file=find_dotenv(), env_file_encoding="utf-8", extra="ignore")

	@property
	def db_dsn(self) -> str:
		return f"postgresql://postgres@/postgres?host={self.pg_socket}"


settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
	load_dotenv()

	# Initialize database connection pool
	print("Connecting to database...")
	pool = await asyncpg.create_pool(settings.db_dsn)
	app.state.pool = pool

	# Load grid classifier model
	print("Loading grid classifier model...")
	app.state.grid_classifier = GridClassifier(768, 0.0, 2)
	app.state.grid_classifier.load_state_dict(torch.load('../../grid-detector/classifier.pt'))
	app.state.grid_classifier.eval()
	app.state.grid_scores = {}

	# Load quality classifier model
	print("Loading quality classifier model...")
	app.state.quality_classifier = QualityClassifier(768, 0.0)
	app.state.quality_classifier.load_state_dict(torch.load('../classifier.pt'))
	app.state.quality_classifier.eval()

	# Precompute dataset
	async with pool.acquire() as conn:
		print("Fetching dataset...")
		app.state.dataset = await fetch_dataset(conn)
		print(f"Dataset loaded with {len(app.state.dataset)} images")
	
		print("Boot-strapping TrueSkill from existing comparisons...")
		app.state.ts_env = ts.TrueSkill(draw_probability=0, sigma=3.0, tau=0.01)
		app.state.ratings, app.state.decile_edges = await build_trueskill(app.state.ts_env, conn, list(app.state.dataset.keys()))
		print(f"TrueSkill ratings bootstrapped for {len(app.state.ratings)} images")

	yield
	await pool.close()


async def get_conn():
	async with app.state.pool.acquire() as conn:
		yield conn


app = FastAPI(lifespan=lifespan)
app.add_middleware(
CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


HexHash = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]


@app.get("/api/images/{filehash}")
async def serve_image(filehash: HexHash, conn: asyncpg.Connection = Depends(get_conn)):
	headers = {
		"Cache-Control": "public, max-age=31536000, immutable",
		"ETag": filehash,
	}

	path = await filehash_to_path(conn, bytes.fromhex(filehash))
	if path is None:
		raise HTTPException(status_code=404, detail="Image not found")
	mime = await asyncio.to_thread(magic.from_file, str(path), mime=True)
	#return StreamingResponse(open(path, "rb"), media_type=mime, headers=headers)
	return FileResponse(path, media_type=mime, headers=headers)


async def fetch_existing_ratings(conn: asyncpg.Connection) -> set[tuple[bytes, bytes]]:
	rows = await conn.fetch('SELECT win_filehash, lose_filehash FROM quality_ratings WHERE source = $1', 'human')
	rows = ((bytes(row[0]), bytes(row[1])) for row in rows)
	return set(((a, b) if a < b else (b, a) for a, b in rows))


async def fetch_dataset(conn: asyncpg.Connection) -> dict[bytes, torch.Tensor]:
	# Sort by filehash and return only the first DATASET_LEN hashes
	# This gives a random, but consistent, subset of the dataset
	# Assuming the dataset doesn't change, all worker processes will get the same subset
	rows = await conn.fetch("SELECT filehash, embedding FROM images WHERE embedding IS NOT NULL ORDER BY filehash LIMIT $1", DATASET_LEN)
	dataset = {}

	for row in rows:
		filehash = bytes(row[0])
		embedding = bytes(row[1])
		embedding_tensor = torch.frombuffer(bytearray(embedding), dtype=torch.float16).to(torch.float32)
		dataset[filehash] = embedding_tensor

	return dataset


async def build_trueskill(env: ts.TrueSkill, conn: asyncpg.Connection, filehashes: list[bytes]) -> tuple[dict[bytes, ts.Rating], list[float]]:
	ratings = {h: env.Rating() for h in filehashes}

	rows = await conn.fetch("SELECT win_filehash, lose_filehash FROM quality_ratings WHERE source = 'human'")
	for win, lose in rows:
		if bytes(win) not in ratings or bytes(lose) not in ratings:
			continue
		w, l = ratings[bytes(win)], ratings[bytes(lose)]
		ratings[bytes(win)], ratings[bytes(lose)] = env.rate_1vs1(w, l)
	
	decile_edges = _compute_deciles(ratings.values())
	return ratings, decile_edges


def _compute_deciles(ratings):
	mus = [r.mu for r in ratings]
	return np.percentile(mus, [10, 20, 30, 40, 50, 60, 70, 80, 90]).tolist()


async def path_to_content(path: Path) -> str:
	image_data = await asyncio.to_thread(path.read_bytes)
	mime = magic.from_buffer(image_data, mime=True)
	return f"data:{mime};base64,{base64.b64encode(image_data).decode('utf-8')}"


@app.get("/api/openai-score/{filehash_a}/{filehash_b}")
async def openai_score(filehash_a: HexHash, filehash_b: HexHash, conn: asyncpg.Connection = Depends(get_conn)):
	client = openai.AsyncOpenAI()
	path_a = await filehash_to_path(conn, bytes.fromhex(filehash_a))
	path_b = await filehash_to_path(conn, bytes.fromhex(filehash_b))
	if path_a is None or path_b is None:
		raise HTTPException(status_code=404, detail="Image not found")
	
	content_a = await path_to_content(path_a)
	content_b = await path_to_content(path_b)

	response = await client.chat.completions.create(
		model="o4-mini",
		#model="gpt-4.1",
		reasoning_effort="medium",
		#temperature=0.5,
		max_completion_tokens=2048,
		response_format={"type": "json_object"},
		messages=[
			{
				"role": "system",
				"content": SYSTEM_PROMPT,
			},
			{
				"role": "user",
				"content": [
					{
						"type": "image_url",
						"image_url": {
							"url": content_a,
							"detail": "high"
						},
					},
					{
						"type": "image_url",
						"image_url": {
							"url": content_b,
							"detail": "high"
						},
					},
					{
						"type": "text",
						"text": USER_PROMPT,
					}
				]
			},
		]
	)

	if len(response.choices) == 0 or response.choices[0].message.content is None:
		print(f"Empty response from OpenAI: {response}")
		raise HTTPException(status_code=500, detail="Empty response from OpenAI")
	
	content = response.choices[0].message.content

	try:
		content = json.loads(content)
	except Exception as e:
		print(f"Failed to parse OpenAI response: {content}: {e}")
		raise HTTPException(status_code=500, detail="Failed to parse OpenAI response")
	
	print(content)

	if 'judgement' in content:
		reasoning = str(content['judgement'])
	else:
		reasoning = ""

	if 'better_image' not in content:
		print("Missing winner in response for")
		raise HTTPException(status_code=500, detail="Missing winner in response")
	
	if content['better_image'].lower().strip() == 'a':
		return {
			"score": 0.0,
			"reasoning": reasoning,
		}
	elif content['better_image'].lower().strip() == 'b':
		return {
			"score": 1.0,
			"reasoning": reasoning,
		}
	
	print(f"Invalid winner in response: {content['better_image']}")
	raise HTTPException(status_code=500, detail="Invalid winner in response")


@app.get("/api/tts")
async def openai_speech(text: str):
	client = openai.AsyncOpenAI()

	response = await client.audio.speech.create(
		model="tts-1-hd",
		#model = "gpt-4o-mini-tts",
		voice="shimmer",
		#voice="nova",
		input=text,
		speed=1.5,
		#instructions="Read quickly and clearly.",
	)

	# Encode in base64 for url
	content = response.content
	content_base64 = base64.b64encode(content).decode('utf-8')

	return Response(content=f"data:audio/mpeg;base64,{content_base64}", media_type="text/plain")


@torch.no_grad()
def compute_grid_score(model: GridClassifier, embedding: torch.Tensor) -> float:
	"""
	Uses the GridClassifier to determine which images in our dataset are likely to be grid images.
	These images should always be ranked lower than non-grid images.
	"""
	logits = model(embedding.unsqueeze(0))
	probabilities = F.softmax(logits, dim=1)[:, 1]  # Get probabilities for the "grid" class
	return probabilities.item()


@torch.no_grad()
def score_image_pair(model: QualityClassifier, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
	"""
	Returns the probability that image2 is of higher quality than image1.
	For example, a score of 0.1 means that image1 is likely to be of higher quality than image2,
	while a score of 0.9 means that image2 is likely to be of higher quality than image1.
	"""
	score = model(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).squeeze()
	score = torch.softmax(score, dim=0)
	assert score.shape == (2,)

	return score[1].item()


def _next_active_pair(ratings: dict[bytes, ts.Rating], edges: list[float], seen_pairs: set[tuple[bytes, bytes]]) -> tuple[bytes, bytes] | None:
	needs = []
	for h, r in ratings.items():
		lo, hi = r.mu - 1.96*r.sigma, r.mu + 1.96*r.sigma
		crossings = sum(lo < e < hi for e in edges)
		if crossings:
			needs.append((crossings * r.sigma, h))
	if not needs:
		return None
	i = max(needs)[1]

	# 2. choose an opponent whose μ is on the opposite side of the closest edge
	target = min((e for e in edges if ratings[i].mu < e),
					default=max(edges))
	j = min(ratings.keys(),
			key=lambda k: abs(ratings[k].mu - target)
			if k != i else float('inf'))

	if (min(i, j), max(i, j)) in seen_pairs:
		return _fallback_unseen_pair(ratings, seen_pairs)  # rare
	return i, j


def _progress_stats(ratings, edges, n_pairs_done):
	sigmas   = np.array([r.sigma for r in ratings.values()])
	unresolved = sum(
		any(r.mu - 1.96*r.sigma < e < r.mu + 1.96*r.sigma for e in edges)
		for r in ratings.values()
	)
	pct_done = 100 * (1 - unresolved / len(ratings))
	info_lb  = len(ratings) * log2(10)          # ≈ N log₂ K
	return {
		"total_ratings": n_pairs_done,
		"avg_sigma":     float(sigmas.mean()),
		"unresolved_items": unresolved,
		"percent_finished": round(pct_done, 2),
		"theoretical_min_pairs": int(info_lb)
	}


def _fallback_unseen_pair(ratings: dict[bytes, ts.Rating],
                          seen_pairs: set[tuple[bytes, bytes]],
                          max_trials: int = 10_000
                         ) -> tuple[bytes, bytes]:
	"""
	Pick any pair of distinct images that has not yet been rated.
	Used only when the actively-chosen pair has already been seen
	(should be rare).
	"""
	keys = list(ratings.keys())
	for _ in range(max_trials):
		a, b = random.sample(keys, 2)
		pair_key = (a, b) if a < b else (b, a)
		if pair_key not in seen_pairs:
			return a, b
	# If we get here, every possible pair has been rated
	raise RuntimeError("No unseen image pairs remain.")


@app.get('/api/random-pair')
async def random_pair(grid: bool = False, conn: asyncpg.Connection = Depends(get_conn)):
	existing_ratings = await fetch_existing_ratings(conn)
	filehashes = list(app.state.dataset.keys())

	pair = _next_active_pair(app.state.ratings, app.state.decile_edges, existing_ratings)
	if pair is None:
		raise HTTPException(status_code=404, detail="No more pairs available")
	
	a, b = pair
	stats = _progress_stats(app.state.ratings, app.state.decile_edges, len(existing_ratings))
	score = score_image_pair(app.state.quality_classifier, app.state.dataset[a], app.state.dataset[b])

	return {
		"image1": a.hex(),
		"image2": b.hex(),
		"score": score,
		**stats
	}

	for _ in range(10000):
		pair = random.sample(filehashes, 2)
		k = tuple(sorted(pair))
		if k in existing_ratings:
			continue  # Skip if this pair has already been rated

		a_embedding = app.state.dataset.get(pair[0])
		b_embedding = app.state.dataset.get(pair[1])

		if grid:
			if pair[0] not in app.state.grid_scores:
				app.state.grid_scores[pair[0]] = compute_grid_score(app.state.grid_classifier, a_embedding)
			if pair[1] not in app.state.grid_scores:
				app.state.grid_scores[pair[1]] = compute_grid_score(app.state.grid_classifier, b_embedding)

			a_is_grid = app.state.grid_scores[pair[0]] > 0.5
			b_is_grid = app.state.grid_scores[pair[1]] > 0.5

			# If both are grid images or both are not grid images, skip
			if (a_is_grid and b_is_grid) or (not a_is_grid and not b_is_grid):
				continue

			# If the classifier isn't confident that the grid image sucks, return the pair
			score = score_image_pair(app.state.quality_classifier, a_embedding, b_embedding)
			if (a_is_grid and score < 0.9) or (b_is_grid and score > 0.1):
				return {
					"image1": pair[0].hex(),
					"image2": pair[1].hex(),
					"total_ratings": len(existing_ratings),
					"score": score,
				}
		else:
			score = score_image_pair(app.state.quality_classifier, a_embedding, b_embedding)

			return {
				"image1": pair[0].hex(),
				"image2": pair[1].hex(),
				"total_ratings": len(existing_ratings),
				"score": score,
			}
	
	raise HTTPException(status_code=404, detail="No more pairs available")


@app.post('/api/rate')
async def rate(winner: HexHash, loser: HexHash, is_tie: bool | None = False, conn: asyncpg.Connection = Depends(get_conn)):
	winner_hash = bytes.fromhex(winner)
	loser_hash = bytes.fromhex(loser)

	if winner_hash == loser_hash:
		raise HTTPException(status_code=400, detail="Winner and loser cannot be the same image")

	row = await conn.fetchval('SELECT COUNT(*) FROM images WHERE filehash = $1 OR filehash = $2', winner_hash, loser_hash)
	assert row is not None and isinstance(row, int), f"Invalid row count: {row}"
	if row != 2:
		raise HTTPException(status_code=400, detail="Both images must exist in the database")
	
	async with conn.transaction():
		await conn.execute('INSERT INTO quality_ratings (win_filehash, lose_filehash, source) VALUES ($1, $2, $3)', winner_hash, loser_hash, 'human')
		if is_tie:
			await conn.execute('INSERT INTO quality_ratings (win_filehash, lose_filehash, source) VALUES ($1, $2, $3)', loser_hash, winner_hash, 'human')
	
	env = app.state.ts_env
	r = app.state.ratings
	r[winner_hash], r[loser_hash] = env.rate_1vs1(r[winner_hash], r[loser_hash])

	# cheap background refresh of decile cut-lines every 50 new votes
	if await conn.fetchval("SELECT COUNT(*) FROM quality_ratings WHERE source = 'human'") % 50 == 0:
		app.state.decile_edges = _compute_deciles(r.values())
	
	return {"status": "success"}


async def filehash_to_path(conn: asyncpg.Connection, filehash: bytes) -> Path | None:
	path = await conn.fetchval('SELECT path FROM images WHERE filehash = $1', filehash)
	assert path is None or isinstance(path, str), f"Invalid path: {path}"
	return Path(path) if path is not None else None
