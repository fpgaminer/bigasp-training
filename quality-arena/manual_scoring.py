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
from grid_models import NsfwClassifier as GridClassifier
import torch.nn.functional as F
from tqdm import tqdm
import psycopg
from pathlib import Path
import openai
from dotenv import load_dotenv
import magic
import json
import base64


DATASET_LEN = 4000


app = Flask(__name__)
g_model = None


mime_to_extension = {
	'image/jpeg': '.jpg',
	'image/png': '.png',
	'image/gif': '.gif',
	'image/bmp': '.bmp',
	'image/tiff': '.tiff',
	'image/webp': '.webp',
}


def fetch_existing_ratings() -> set[tuple[str, str]]:
	with psycopg.connect(dbname='postgres', user='postgres', host=str(Path.cwd().parent / "pg-socket")) as conn:
		cursor = conn.cursor()
		cursor.execute('SELECT win_path, lose_path FROM quality_ratings WHERE source = %s', ('human',))
		return set(cursor.fetchall())


def fetch_dataset_paths() -> list[str]:
	with psycopg.connect(dbname='postgres', user='postgres', host=str(Path.cwd().parent / "pg-socket")) as conn:
		cursor = conn.cursor()
		cursor.execute('SELECT path FROM images WHERE embedding IS NOT NULL')
		all_paths = [row[0] for row in cursor.fetchall()]

	existing_ratings = fetch_existing_ratings()
	dataset_paths = set(itertools.chain(*existing_ratings))

	while len(dataset_paths) < DATASET_LEN:
		dataset_paths.add(random.choice(all_paths))
	
	return list(dataset_paths)


def path_to_content(client: openai.Client, path: Path):
	try:
		# Identify image
		file_type = magic.from_file(str(path), mime=True)

		# Upload image
		file_name = path.with_suffix(mime_to_extension[file_type]).name
		file = client.files.create(file=(file_name, open(path, 'rb'), file_type), purpose='vision') # type: ignore
	except Exception as e:
		print(f"Failed to upload image {path}: {e}")
		return None, None
	
	content = {
		"type": "image_file",
		"image_file": {"file_id": file.id, "detail": "high"},
	}

	return content, file


def openai_score_pair(path: str, other_path: str) -> tuple[float, str]:
	load_dotenv()
	client = openai.Client()
	assistant = client.beta.assistants.retrieve('')

	image_content, file = path_to_content(client, Path(path))
	if image_content is None:
		return 0.5, ""

	content = [image_content]
	files = [file]

	image_content, file = path_to_content(client, Path(other_path))
	if image_content is None:
		return 0.5, ""
	
	content.append(image_content)
	files.append(file)

	content.append({
		"type": "text",
		"text": "Please compare these two images and tell me which one you think is of higher subjective quality. Compare based on their quality within their respective field. For example, and photo is not automatically better than a drawing; it would only be better if the drawing was of poor quality for a drawing and the photo was of high quality for a photo. Please be sure to also consider the presence of jpeg artifacts, etc which are generally bad. Respond with a JSON object containing the key 'reasoning' with a string where you consider the factors about each image. And a key 'winner' with 1 for the first image and 2 for the second image indicating which one you think is better. You MUST respond with a JSON object with these keys, and you MUST respond with a winner no matter how close the decision is.",
	})

	# Create thread
	thread = client.beta.threads.create(messages=[{"role": "user", "content": content}])
	print(f"Created thread {thread.id}")

	# Run completion
	run = client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id=assistant.id)
	print(f"Completed run {run.id}, {run}")

	# Check
	if run.status != 'completed':
		print(f"Run status is {run.status} for image {path}, thread ID {thread.id}, {run.id}")
		return 0.5, ""
	
	# Get messages
	messages = client.beta.threads.messages.list(thread_id=thread.id)

	if len(messages.data) < 2:
		print(f"Less than 2 messages in thread {thread.id}")
		return 0.5, ""
	
	message = messages.data[0]
	if message.role != 'assistant':
		print(f"Message is not from assistant in thread {thread.id}")
		return 0.5, ""
	
	if len(message.content) != 1:
		print(f"Message content is not 1 in thread {thread.id}")
		return 0.5, ""
	
	if message.content[0].type != 'text':
		print(f"Message content type is not text in thread {thread.id}")
		return 0.5, ""
	
	response = message.content[0].text.value

	# Cleanup
	try:
		client.beta.threads.delete(thread.id)
		for file in files:
			client.files.delete(file.id)
	except Exception as e:
		print(f"Failed to cleanup thread and file for image {path}: {e}")

	try:
		response = json.loads(response)
	except Exception as e:
		print(f"Failed to parse response for image {path}: {e}")
		return 0.5, ""
	
	print(response)

	if 'reasoning' in response:
		reasoning = str(response['reasoning'])
	else:
		reasoning = ""

	if 'winner' not in response:
		print(f"Missing winner in response for image {path}")
		return 0.5, ""
	
	if response['winner'] == 1:
		return 0.0, reasoning
	elif response['winner'] == 2:
		return 1.0, reasoning
	
	print(f"Invalid winner in response for image {path}")

	return 0.5, reasoning


def openai_speech(text: str) -> str:
	load_dotenv()

	client = openai.Client()

	response = client.audio.speech.create(
		model="tts-1-hd",
		voice="shimmer",
		input=text,
		speed=1.5,
	)

	# Encode in base64 for url
	content = response.content
	content_base64 = base64.b64encode(content).decode('utf-8')

	return f"data:audio/mpeg;base64,{content_base64}"



@torch.no_grad()
def compute_grid_scores():
	"""
	Uses the GridClassifier to determine which images in our dataset are likely to be grid images.
	These images should always be ranked lower than non-grid images.
	"""
	global g_grid_scores

	# Use trained classifier to score images
	model = GridClassifier(768, 0.0, 2)
	#model.load_state_dict(torch.load('../watermark-detector/classifier.pt'))
	model.load_state_dict(torch.load('../grid-detector/classifier.pt'))
	model.eval()

	g_grid_scores = {}

	with psycopg.connect(dbname='postgres', user='postgres', host=str(Path.cwd().parent / "pg-socket")) as conn:
		cursor = conn.cursor()

		for path in tqdm(dataset_paths, desc='Classifying grid images'):
			cursor.execute('SELECT embedding FROM images WHERE path = %s', (path,))
			embedding = cursor.fetchone()
			if embedding is None or embedding[0] is None:
				raise ValueError(f"Missing embedding for {path}")
			embedding = bytes(embedding[0])
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
	
	with psycopg.connect(dbname='postgres', user='postgres', host=str(Path.cwd().parent / "pg-socket")) as conn:
		cursor = conn.cursor()
		cursor.execute('SELECT embedding FROM images WHERE path = %s', (image1,))
		embedding1 = bytes(cursor.fetchone()[0])
		embedding1 = torch.frombuffer(bytearray(embedding1), dtype=torch.float16).to(torch.float32)
		cursor.execute('SELECT embedding FROM images WHERE path = %s', (image2,))
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
	#openai_score, openai_reason = openai_score_pair(pair[0], pair[1])
	openai_score = 0.5
	openai_reason = ""

	if openai_reason != "":
		if openai_score == 0.0:
			winner_text = "the first image"
		elif openai_score == 1.0:
			winner_text = "the second image"
		else:
			winner_text = "neither image"
		speech = openai_speech(f"The winner is {winner_text}. " + openai_reason)
	else:
		speech = ""
	
	return render_template('index.html', image1=pair[0], image2=pair[1], stats=stats, ai_score=ai_score, grid=get_grid, openai_score=openai_score, openai_reason=openai_reason, speech=speech)


@app.route('/rate', methods=['POST'])
def rate():
	winner = request.form['winner']
	loser = request.form['loser']
	assert winner in dataset_paths
	assert loser in dataset_paths
	assert winner != loser

	with psycopg.connect(dbname='postgres', user='postgres', host=str(Path.cwd().parent / "pg-socket")) as conn:
		cursor = conn.cursor()
		cursor.execute('INSERT INTO quality_ratings (win_path, lose_path, source) VALUES (%s, %s, %s)', (winner, loser, 'human'))
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

	with psycopg.connect(dbname='postgres', user='postgres', host=str(Path.cwd().parent / "pg-socket")) as conn:
		cursor = conn.cursor()
		cursor.execute('INSERT INTO quality_ratings (win_path, lose_path, source) VALUES (%s, %s, %s)', (image1, image2, 'human'))
		cursor.execute('INSERT INTO quality_ratings (win_path, lose_path, source) VALUES (%s, %s, %s)', (image2, image1, 'human'))
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