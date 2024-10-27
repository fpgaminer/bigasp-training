#!/usr/bin/env python3
"""
Go through the checkpoints and remove unnecessary files.
The most recent two checkpoints are kept completely intact.
Earlier checkpoints have all of their training state removed.
Only one checkpoint per million samples is kept.
"""
from pathlib import Path
import argparse
import dataclasses
import shutil


@dataclasses.dataclass(frozen=True)
class Checkpoint:
	path: Path
	n_samples: int
	training_files: tuple[Path, ...]


@dataclasses.dataclass
class Run:
	path: Path
	checkpoints: list[Checkpoint]


parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="checkpoints", help="The folder to clean.")
parser.add_argument("--dry-run", action="store_true", default=False, help="Don't actually delete anything.")


def main():
	args = parser.parse_args()

	# Find all runs
	runs = find_runs(Path(args.folder))

	# Clean each run
	for run in runs:
		clean_run(run, args.dry_run)



def clean_run(run: Run, dry_run: bool):
	# Sort the checkpoints by number of samples from smallest to largest
	run.checkpoints.sort(key=lambda x: x.n_samples)

	# Keep the most recent two
	checkpoints = run.checkpoints[:-2]

	# Only keep checkpoints every million samples
	n_samples = 0
	keep = set()

	while len(checkpoints) > 0:
		# Find closest checkpoint
		closest = min(checkpoints, key=lambda x: abs(x.n_samples - n_samples))
		keep.add(closest)
		n_samples += 1_000_000

		if checkpoints[-1] == closest:
			break
	
	for checkpoint in checkpoints:
		if checkpoint in keep:
			continue

		print(f"Removing {checkpoint.path}")
		if not dry_run:
			shutil.rmtree(checkpoint.path)

	# Remove all training state from the remaining checkpoints
	for checkpoint in keep:
		for file in checkpoint.training_files:
			print(f"Removing {file}")
			if not dry_run:
				file.unlink()



def find_runs(parent_folder: Path) -> list[Run]:
	runs: list[Run] = []

	for folder in parent_folder.iterdir():
		if not folder.is_dir():
			continue

		# It's a run folder if there are checkpoints in it.
		checkpoints = find_checkpoints(folder)

		if len(checkpoints) == 0:
			continue

		runs.append(Run(folder, checkpoints))
	
	return runs


def find_checkpoints(run_folder: Path) -> list[Checkpoint]:
	checkpoints: list[Checkpoint] = []

	for folder in run_folder.iterdir():
		# Checkpoints are folders with the name samples_*
		if not folder.is_dir() or not folder.name.startswith('samples_'):
			continue

		# Find the number of samples
		try:
			n_samples = int(folder.name[len('samples_'):])
		except ValueError:
			continue

		# Find all training files
		training_files = list(folder.rglob('training_state*.pt'))

		checkpoints.append(Checkpoint(folder, n_samples, tuple(training_files)))
	
	return checkpoints


if __name__ == "__main__":
	main()