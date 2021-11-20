import json
import os
from collections import Counter


dir = "../data/SiwisFrenchSpeechSynthesisDatabase/text/part1"
paths = [os.path.join(dir, x) for x in os.listdir(dir)]


def process(path):
    stream = os.popen(f"perl ./phonetize.pl {path} texts hts run")
    lines = map(lambda l: l.strip(), stream
                .read()
                .replace("mode=run\n", "¤")
                .replace("\nDELETE", "¤")
                .split("¤")[1:-1][0].split())

    phonemes = []
    for x in lines:
        splt = x.replace("-", "¤").replace("+", "¤").split("¤")
        if len(splt) > 1:
            phonemes.append(splt[1])

    return phonemes


output = list(map(lambda file: process(file), paths))

flat = [xi for x in output for xi in x]
phon2cnt = dict(Counter(flat).most_common())
phon2idx = {x: i for i, x in enumerate(phon2cnt.keys())}

dict_path = "../data/phon2idx"
with open(dict_path, "w") as f:
    json.dump(phon2idx, f)

# reading
# with open(dict_path, "r") as f:
#     phon2cnt = json.load(f)

idx2phon = {v: k for k, v in phon2idx.items()}

with open(os.path.join(os.path.dirname(os.path.abspath(dir)), "../data/output.txt"), "w") as fout:
    for path, phonemes in zip(paths, output):
        line = f"{path} | {' '.join([str(phon2idx[x]) for x in phonemes])}\n"
        print(line.split("/")[-1])
        fout.write(line)
