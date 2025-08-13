import torch.nn as nn
import torch


# class NsfwClassifier(nn.Module):
# 	def __init__(self, embedding_size: int, dropout: float, outputs: int):
# 		super(NsfwClassifier, self).__init__()

# 		self.ln = nn.LayerNorm(embedding_size)
# 		self.dropout1 = nn.Dropout(dropout)
# 		self.linear1 = nn.Linear(embedding_size, outputs)
	
# 	def forward(self, x):
# 		x = self.ln(x)
# 		x = self.dropout1(x)
# 		x = self.linear1(x)
# 		return x


class NsfwClassifier(nn.Module):
	def __init__(self, embedding_size: int, dropout: float, outputs: int):
		super(NsfwClassifier, self).__init__()

		self.ln = nn.LayerNorm(embedding_size)
		self.dropout1 = nn.Dropout(dropout)
		self.linear1 = nn.Linear(embedding_size, embedding_size*2)
		self.act_fn = nn.GELU()
		self.dropout2 = nn.Dropout(dropout)
		self.linear2 = nn.Linear(embedding_size*2, outputs)
	
	def forward(self, x: torch.Tensor):
		x = self.ln(x)
		x = self.dropout1(x)
		x = self.linear1(x)

		x = self.act_fn(x)
		x = self.dropout2(x)
		x = self.linear2(x)

		return x