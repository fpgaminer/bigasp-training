CREATE TABLE IF NOT EXISTS ratings (
	win_path TEXT NOT NULL,
	lose_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gemini_ratings (
	win_path TEXT NOT NULL,
	lose_path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gpt4v2_ratings (
	win_path TEXT NOT NULL,
	lose_path TEXT NOT NULL
);