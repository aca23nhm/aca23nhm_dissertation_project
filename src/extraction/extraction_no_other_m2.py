import os
from tqdm import tqdm

M2_DIR = "data\\bea2019\\m2"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def extract_sentences_no_other(m2_path):
    """
    Reads an M2 file and returns sentences where NONE of the edits have OTHER.
    Keeps all A-lines for that sentence.
    """
    output_lines = []
    with open(m2_path, "r", encoding="utf8") as f:
        current_sentence = None
        edits = []

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("S "):
                # Process previous sentence
                if current_sentence and all("OTHER" not in e for e in edits):
                    output_lines.append(f"S {current_sentence}")
                    output_lines.extend(edits)
                    output_lines.append("")  # blank line between sentences

                current_sentence = line[2:]
                edits = []
            elif line.startswith("A "):
                edits.append(line)

        # Handle last sentence
        if current_sentence and all("OTHER" not in e for e in edits):
            output_lines.append(f"S {current_sentence}")
            output_lines.extend(edits)
            output_lines.append("")

    return output_lines


# Process each .m2 file individually
for file in tqdm(os.listdir(M2_DIR)):
    if file.endswith(".m2"):
        file_path = os.path.join(M2_DIR, file)
        filtered_lines = extract_sentences_no_other(file_path)
        if filtered_lines:
            output_file = os.path.join(RESULTS_DIR, f"{os.path.splitext(file)[0]}_no_other.m2")
            with open(output_file, "w", encoding="utf8") as out_f:
                out_f.write("\n".join(filtered_lines))
            print(f" {output_file} — {len(filtered_lines)} lines written")
