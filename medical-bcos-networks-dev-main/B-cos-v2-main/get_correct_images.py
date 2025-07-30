import json
import argparse

parser = argparse.ArgumentParser(
)
parser.add_argument(
    "--files",
    type=str,
    nargs="+"
)
parser.add_argument(
    "--outname",
    type=str,
)

args = parser.parse_args()

all_data = []
for f in args.files:
    with open(f, "r") as f:
        data = json.load(f)
    all_data.append(data)

intersect = list(set.intersection(*map(set, all_data)))
with open(args.outname, "w") as f:
    json.dump(intersect, f, indent=4)
