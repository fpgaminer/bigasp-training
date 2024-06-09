#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


parser = argparse.ArgumentParser(description="Parallel rsync folders to a remote host.")
parser.add_argument("--source", help="Source directory containing folders to rsync")
parser.add_argument("--dest", help="Destination directory on the remote host")
parser.add_argument("--host", help="Destination host in the format user@host")
parser.add_argument("--port", type=int, default=22, help="SSH port on the destination host")
parser.add_argument("--parallel", type=int, default=4, help="Number of parallel rsync operations")


def main():
	args = parser.parse_args()
	args.source = Path(args.source)

	if not args.source.exists() or not args.source.is_dir():
		raise ValueError(f"Source directory {args.source} does not exist or is not a directory")
	
	# List all folders in the source
	src_folders = [f for f in args.source.iterdir() if f.is_dir()]

	with ThreadPoolExecutor(max_workers=args.parallel) as executor:
		futures = [executor.submit(rsync_folder, src_folder, args.dest, args.host, args.port) for src_folder in src_folders]
		for future in futures:
			future.result()


def rsync_folder(src_folder: Path, dest: str, host: str, port: int):
	print(f"rsync -av -e ssh -p {port} {src_folder} {host}:{dest}")
	subprocess.run([
		"rsync",
		"-a",
		"--info=progress2",
		"-e",
		f"ssh -p {port}",
		str(src_folder),
		f"{host}:{dest}"
	], check=True)


if __name__ == "__main__":
	main()