import { RandomPairResponse } from '../types'

/**
 * Fetches a random pair of images from the backend
 * @param gridMode Whether to fetch a pair that includes a grid image
 */
export async function fetchRandomPair(gridMode: boolean = false): Promise<RandomPairResponse> {
  const response = await fetch(`/api/random-pair?grid=${gridMode ? 'true' : 'false'}`)
  if (!response.ok) {
    throw new Error(`Failed to fetch random pair: ${response.status}`)
  }
  return response.json()
}

/**
 * Submits a rating where one image is selected as better than the other
 * @param winner The hash of the winning image
 * @param loser The hash of the losing image
 * @param isTie Whether this should be recorded as a tie (both images rated equally)
 */
export async function submitRating(winner: string, loser: string, isTie: boolean = false): Promise<void> {
  // Create URL with search parameters that FastAPI can parse
  const url = new URL('/api/rate', window.location.origin);
  url.searchParams.append('winner', winner);
  url.searchParams.append('loser', loser);
  if (isTie) {
    url.searchParams.append('is_tie', 'true');
  }
  
  const response = await fetch(url, {
    method: 'POST',
  });
  
  if (!response.ok) {
    throw new Error(`Failed to submit rating: ${response.status}`)
  }
}

/**
 * Fetches the OpenAI evaluation of the image pair
 * @param image1 The hash of the first image
 * @param image2 The hash of the second image
 */
export async function fetchOpenAIScore(image1: string, image2: string): Promise<{
  score: number;
  reasoning: string;
}> {
  const response = await fetch(`/api/openai-score/${image1}/${image2}`)
  if (!response.ok) {
    throw new Error(`Failed to fetch OpenAI score: ${response.status}`)
  }
  return response.json()
}

/**
 * Gets the URL for an image by its hash
 * @param imageHash The hash of the image
 */
export function getImageUrl(imageHash: string): string {
  return `/api/images/${imageHash}`
}

/**
 * Gets text-to-speech audio for the provided text
 * @param text The text to convert to speech
 */
export async function getTextToSpeech(text: string): Promise<string> {
  // URL encode the text as a query parameter
  const encodedText = encodeURIComponent(text);
  const response = await fetch(`/api/tts?text=${encodedText}`)
  if (!response.ok) {
    throw new Error(`Failed to fetch speech: ${response.status}`)
  }
  return response.text()
}