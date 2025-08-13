# Quality Arena

I manually rate pairs of images, deciding which of the two is better. In terms of aesthetics, composition, and overall quality. Divorced from content.
Those ratings are added to the database as win_path, lose_path.
A tie is added to the database as two entries, with each image winning once.


To perform manual rating work, run:
* uvicorn backend:app --port 5034 --reload
* npm run dev
That will bring the web app up.


A model can then be trained off of these ratings. The model will predict which image is better, given two images as input using CLIP embeddings.

From that, the model is used to build a large dataset of images, ranked using OpenSkill. This dataset can then be used to train a model to predict an image's overall score, from [0, 9].




## How to train the ranking model.
This model predicts which of two given images is higher quality.
Use `TrainClassifier.ipynb` to train the model.


## How to train the scoring model
This model predicts the quality of a single image on a scale of [0, 9].
Use `TrainRanker-OpenSkill.ipynb` to run OpenSkill on a dataset using the trained classifier model.
Use `TrainRanker-OpenSkill-2.ipynb` to train the scoring model on the OpenSkill dataset.
