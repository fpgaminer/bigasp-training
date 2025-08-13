export interface ImagePair {
  image1: string;
  image2: string;
}

export interface OpenAIScoreResult {
  score: number;
  reasoning: string;
}

export interface RandomPairResponse {
  image1: string;
  image2: string;
  score: number;
  total_ratings: number;
  avg_sigma: number;
  unresolved_items: number;
  percent_finished: number;
  theoretical_min_pairs: number;
}