# Quality Arena

This is a dump of the scripts and notebooks used to train the quality/scoring models.  It's an absolute mess, so caveat emptor.  Some files were moved around after the fact, so paths/loaded models/etc might be incorrect in the current code/notebooks.





To perform manual rating work, run `./manual_scoring.py` and open the URL in a browser.  Click the better image, or click "Tie" to rank the images as a tie.  To actually skip, just reload the page.


A model can then be trained off of these ratings. The model will predict which image is better, given two images as input using CLIP embeddings.

From that, the model is used to build a large dataset of images, ranked using ELO. This dataset can then be used to train a model to predict an image's overall score, from [0, 9] where 0 gets excluded from training the generative model.




## How to train the ranking model.
This model predicts which of two given images is higher quality.
Use `TrainClassifier.ipynb` to train the model.


## How to train the scoring model
This model predicts the quality of a single image on a scale of [0, 9].
Use `TrainRanker.ipynb` to train the model.


## Score all images
`score_images.py` will score all images in the database using the scoring model.