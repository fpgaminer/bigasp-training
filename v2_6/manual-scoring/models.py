import torch.nn as nn
import torch


class QualityClassifier(nn.Module):
	"""
	Predicts which of the two input images is of higher quality.
	"""

	def __init__(self, embedding_size, dropout):
		super(QualityClassifier, self).__init__()

		self.ln = nn.LayerNorm(embedding_size)
		self.seq = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(embedding_size * 2, embedding_size * 4),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(embedding_size * 4, 2)
		)
		#self.dropout1 = nn.Dropout(dropout)
		#self.linear1 = nn.Linear(embedding_size * 2, 2)
	
	def forward(self, a: torch.Tensor, b: torch.Tensor):
		a = self.ln(a)
		b = self.ln(b)
		x = torch.cat([a, b], dim=1)
		assert x.shape[1] == a.shape[1] * 2 and x.shape[0] == a.shape[0] and len(x.shape) == 2

		return self.seq(x)

		#x = self.dropout1(x)
		#x = self.linear1(x)

		#return x


class ScoreClassifier(nn.Module):
	def __init__(self, embedding_size: int, dropout: float, bins: int):
		super(ScoreClassifier, self).__init__()

		self.ln = nn.LayerNorm(embedding_size)
		self.dropout1 = nn.Dropout(dropout)
		self.linear1 = nn.Linear(embedding_size, embedding_size*2)
		self.act_fn = nn.GELU()
		self.dropout2 = nn.Dropout(dropout)
		self.linear2 = nn.Linear(embedding_size*2, bins)
	
	def forward(self, x: torch.Tensor):
		x = self.ln(x)
		x = self.dropout1(x)
		x = self.linear1(x)

		x = self.act_fn(x)
		x = self.dropout2(x)
		x = self.linear2(x)

		return x