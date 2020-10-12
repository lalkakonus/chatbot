scene_delimiter = "NEW_SCENE"


def get_all_characters():
    MAX_LENGTH = 80
    characters = set()
    with open("../data/scrubs/raw_transcripts.tsv", "r") as transcripts_file:
        for line in transcripts_file:
            character = line.strip().split("\t")[1]
            if len(character[1]) < MAX_LENGTH:
                characters.add(character)

    with open("../data/scrubs/all_characters.txt", "w") as characters_file:
        for character in characters:
            characters_file.write("{}\n".format(character))


def filter_transcripts():
    with open("../data/scrubs/verified_characters.txt", "r") as characters_file:
        verified_characters = {character.strip() for character in characters_file}

    transcripts = [[0, scene_delimiter, scene_delimiter], ]
    with open("../data/scrubs/raw_transcripts.tsv", "r") as raw_transcripts_file:
        for replica in raw_transcripts_file:
            separated = replica.strip().split("\t")
            if len(separated) < 3:
                continue
            _, character, replica = separated
            if transcripts[-1][1] == scene_delimiter == character or character not in verified_characters:
                continue
            if transcripts[-1][1] == character:
                transcripts[-1][2] += replica
            else:
                transcripts.append([len(transcripts), character, replica])

    with open("../data/scrubs/filtered_transcripts.tsv", "w") as filtered_transcripts_file:
        for line in transcripts:
            filtered_transcripts_file.write("{}\t{}\t{}\n".format(*line))


if __name__ == "__main__":
    filter_transcripts()