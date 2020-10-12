scene_delimiter = "NEW_SCENE"


def validate_conversation(conversation, character):
    for line in conversation[::-2]:
        if character not in line[1]:
            return False
    return True


def build_dataset(character="J.D.", context_length=6):
    data = []
    with open("../data/scrubs/filtered_transcripts.tsv", "r") as filtered_transcripts_file:
        scene = []
        for line in filtered_transcripts_file:
            line = line.strip().split("\t")
            if line[1] == scene_delimiter:
                if len(scene) > 0:
                    data.append(scene)
                    scene = []
            else:
                scene.append(line)

    dataset = []
    for scene in data:
        if len(scene) < context_length + 1:
            continue
        for pos in range(context_length, len(scene)):
            conversation = scene[pos: pos + context_length + 1]
            if pos + context_length + 1 <= len(scene) and validate_conversation(conversation, character):
                dialog = [line[2] for line in conversation]
                dataset.append(dialog)
    return dataset


def save_dataset(dataset, character, context_length):
    with open("../data/scrubs/{}_{}_dataset.tsv".format(character, context_length), "w") as out_f:
        header = "\t".join(["respond", ] + ["context_{}".format(i) for i in range(context_length)]) + "\n"
        out_f.write(header)
        for sample in dataset:
            assert len(sample) == context_length + 1
            out_f.write("\t".join(sample[::-1]) + "\n")


if __name__ == "__main__":
    character = "J.D."
    context_length = 4
    dataset = build_dataset(character, context_length)
    save_dataset(dataset, character, context_length)