import glob
import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, help="RL Environment")
args = parser.parse_args()

files = glob.glob("algorithms/finetune/checkpoints/*")
newest_to_oldest = sorted(files, key=os.path.getctime, reverse=True)

for folder in newest_to_oldest:
    folder_files = glob.glob(f"{folder}/*.pt")
    if args.env in folder and f"{folder}/checkpoint_999999.pt" in folder_files:
        print(f"Moving folder {folder}")
        shutil.copytree(folder, folder[:-8]+"offline")
        break
    

