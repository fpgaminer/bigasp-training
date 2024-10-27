CREATE TABLE IF NOT EXISTS images (
	path TEXT PRIMARY KEY NOT NULL,
	embedding BYTEA,
	joytag_embedding BYTEA,
	score INTEGER,
	caption_type TEXT,
	caption TEXT,
	caption_2 TEXT,         -- After caption has been run through the recaptioner
	caption_3 TEXT,         -- After caption_2 has been run through the watermark recaptioner
	caption_4 TEXT,         -- After caption_3 has been run through the add source recaptioner
	caption_metadata JSONB,
	subreddit TEXT,
	watermark BOOLEAN,
	watermark_boxes BYTEA,
	source TEXT,
	username TEXT,
	tag_string TEXT
);


CREATE TABLE IF NOT EXISTS quality_ratings (
	win_path TEXT NOT NULL,
	lose_path TEXT NOT NULL,
	source TEXT NOT NULL    -- 'human', etc.
);


CREATE TABLE IF NOT EXISTS recaption_dataset (
	caption TEXT NOT NULL,
	recaptioned TEXT NOT NULL,
	operator TEXT NOT NULL,
	extra_1 TEXT,    -- Currently this is used for the source recaptioner
);