#!/usr/bin/env python
"""
# Manual Image Head to Head Scoring
This builds a dataset that we can then hopefully use ELO on to eventually score images.
"""
from PIL import Image
import sqlite3
import itertools
import random
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, abort
import os
import torch
from models import QualityClassifier
from watermark_models import NsfwClassifier as GridClassifier
import torch.nn.functional as F
from tqdm import tqdm


DATASET_LEN = 2200


app = Flask(__name__)
g_model = None


def fetch_existing_ratings() -> set[tuple[str, str]]:
	with sqlite3.connect('ratings.sqlite3') as conn:
		cursor = conn.cursor()
		cursor.execute('SELECT win_path, lose_path FROM ratings')
		return set(cursor.fetchall())


def fetch_dataset_paths() -> list[str]:
	with sqlite3.connect('../data/clip-embeddings.sqlite3') as conn:
		cursor = conn.cursor()
		cursor.execute('SELECT path FROM images WHERE embedding IS NOT NULL')
		all_paths = [row[0] for row in cursor.fetchall()]

	existing_ratings = fetch_existing_ratings()
	dataset_paths = set(itertools.chain(*existing_ratings))

	while len(dataset_paths) < DATASET_LEN:
		dataset_paths.add(random.choice(all_paths))
	
	return list(dataset_paths)


@torch.no_grad()
def compute_grid_scores():
	"""
	Uses the GridClassifier to determine which images in our dataset are likely to be grid images.
	These images should always be ranked lower than non-grid images.
	"""
	global g_grid_scores

	# Use trained classifier to score images
	model = GridClassifier(768, 0.0, 2)
	model.load_state_dict(torch.load('../watermark-detector/classifier.pt'))
	model.eval()

	g_grid_scores = {}

	with sqlite3.connect('../data/clip-embeddings.sqlite3') as conn:
		cursor = conn.cursor()

		for path in tqdm(dataset_paths, desc='Classifying grid images'):
			cursor.execute('SELECT embedding FROM images WHERE path = ?', (path,))
			embedding = bytes(cursor.fetchone()[0])
			embedding = torch.frombuffer(bytearray(embedding), dtype=torch.float16)
			embedding = embedding.to(torch.float32)
			embedding = embedding.unsqueeze(0)

			logits = model(embedding)
			probabilities = F.softmax(logits, dim=1)

			g_grid_scores[path] = probabilities[:, 1].item()


@torch.no_grad()
def score_image_pair(image1: str, image2: str) -> float:
	"""
	Returns the probability that image2 is of higher quality than image1.
	For example, a score of 0.1 means that image1 is likely to be of higher quality than image2,
	while a score of 0.9 means that image2 is likely to be of higher quality than image1.
	"""
	global g_model
	if g_model is None:
		g_model = QualityClassifier(768, 0.0)
		g_model.load_state_dict(torch.load('classifier.pt'))
		g_model.eval()
	
	with sqlite3.connect('../data/clip-embeddings.sqlite3') as conn:
		cursor = conn.cursor()
		cursor.execute('SELECT embedding FROM images WHERE path = ?', (image1,))
		embedding1 = bytes(cursor.fetchone()[0])
		embedding1 = torch.frombuffer(bytearray(embedding1), dtype=torch.float16).to(torch.float32)
		cursor.execute('SELECT embedding FROM images WHERE path = ?', (image2,))
		embedding2 = bytes(cursor.fetchone()[0])
		embedding2 = torch.frombuffer(bytearray(embedding2), dtype=torch.float16).to(torch.float32)
	
	score = g_model(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).squeeze()
	score = torch.softmax(score, dim=0)
	assert score.shape == (2,)

	return score[1].item()


def get_random_grid_pair() -> tuple[str, str] | None:
	"""
	Find a pair of images that is a (assumed) grid image vs a non-grid image where
	the current classifier is scoring the grid image higher, or as a tie.
	"""
	pairs = list(itertools.combinations(dataset_paths, 2))
	random.shuffle(pairs)

	for pair in pairs:
		# Skip if this pair has already been rated
		if pair in existing_ratings or (pair[1], pair[0]) in existing_ratings:
			continue

		a_is_grid = g_grid_scores[pair[0]] > 0.5
		b_is_grid = g_grid_scores[pair[1]] > 0.5

		# If both are grid images or both are not grid images, skip
		if (a_is_grid and b_is_grid) or (not a_is_grid and not b_is_grid):
			continue

		# If the classifier isn't confident that the grid image sucks, return the pair
		score = score_image_pair(pair[0], pair[1])
		if (a_is_grid and score < 0.9) or (b_is_grid and score > 0.1):
			return pair

	return None


def get_random_pair() -> tuple[str, str] | None:
	pairs = list(itertools.combinations(dataset_paths, 2))
	random.shuffle(pairs)

	for pair in pairs:
		if pair not in existing_ratings and (pair[1], pair[0]) not in existing_ratings:
			return pair
	
	return None


@app.route('/')
def index():
	get_grid = request.args.get('grid', default=0, type=int)
	if get_grid:
		pair = get_random_grid_pair()
		if pair is None:
			pair = get_random_pair()
	else:
		pair = get_random_pair()
	
	if pair is None:
		return "No more unique pairs available"
	
	# Calculate total_ratings by filtering out duplicates from the existing_ratings set
	total_ratings = len(set((a,b) if a < b else (b,a) for a,b in existing_ratings))
	
	max_ratings = len(list(itertools.combinations(dataset_paths, 2)))
	stats = f"Total ratings: {total_ratings}, Max ratings: {max_ratings}"

	ai_score = score_image_pair(pair[0], pair[1])
	
	return render_template('index.html', image1=pair[0], image2=pair[1], stats=stats, ai_score=ai_score, grid=get_grid)


@app.route('/rate', methods=['POST'])
def rate():
	winner = request.form['winner']
	loser = request.form['loser']
	assert winner in dataset_paths
	assert loser in dataset_paths
	assert winner != loser

	with sqlite3.connect('ratings.sqlite3') as conn:
		cursor = conn.cursor()
		cursor.execute('INSERT INTO ratings (win_path, lose_path) VALUES (?, ?)', (winner, loser))
		conn.commit()
	
	existing_ratings.add((winner, loser))
	return redirect(url_for('index', grid=request.form.get('grid', default=0, type=int)))


@app.route('/skip', methods=['POST'])
def skip():
	# Record as a tie
	image1 = request.form['image1']
	image2 = request.form['image2']
	assert image1 in dataset_paths
	assert image2 in dataset_paths

	with sqlite3.connect('ratings.sqlite3') as conn:
		cursor = conn.cursor()
		cursor.execute('INSERT INTO ratings (win_path, lose_path) VALUES (?, ?)', (image1, image2))
		cursor.execute('INSERT INTO ratings (win_path, lose_path) VALUES (?, ?)', (image2, image1))
		conn.commit()

	existing_ratings.add((image1, image2))
	existing_ratings.add((image2, image1))
	return redirect(url_for('index', grid=request.form.get('grid', default=0, type=int)))


@app.route('/images/<path:path>')
def serve_image(path):
	if not path.startswith('/'):
		path = '/' + path

	if path not in dataset_paths:
		print(f"Invalid path: {path}")
		abort(404)
	else:
		directory = os.path.dirname(path)
		image = os.path.basename(path)
		print(f"Serving {image} from {directory}")
		return send_from_directory(directory, image)


if __name__ == '__main__':
	dataset_paths = fetch_dataset_paths()
	existing_ratings = fetch_existing_ratings()
	compute_grid_scores()
	app.run(debug=True, port=5034)