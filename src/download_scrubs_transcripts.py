import requests
from bs4 import BeautifulSoup
from typing import List
from tqdm import tqdm
import re


square_braces = re.compile(r"\[[^\]]*\]")
curvy_braces = re.compile(r"\{[^\}]*\}")
round_braces = re.compile(r"\([^\)]*\)")
dashes = re.compile(r"-{2,}")
multy_space = re.compile(r" +")
scene_delimiter_pattern = re.compile(r".*(act|scene).*")
scene_delimiter = "NEW_SCENE"


def get_raw_text(link: str) -> List[str]:
    raw_text = []
    try:
        html = requests.get(link).text
        soup = BeautifulSoup(html, 'html.parser')
        html_text = soup.find(**{"class": "mw-parser-output"})
        for p in html_text.find_all(re.compile("(p|h\2)")):
            raw_text.append(p.text.strip())
    except Exception as error:
        print("Error during process document from {}:\n{}\n".format(link, error))
    return raw_text


def get_raw_transcripts(link: str, speech_id: int) -> List[List[str]]:
    raw_text = get_raw_text(link)
    raw_transcripts = [[speech_id, scene_delimiter, scene_delimiter], ]
    speech_id += 1
    for item in raw_text:
        item = re.sub(square_braces, "", item)
        item = re.sub(curvy_braces, "", item)
        item = re.sub(round_braces, "", item)
        item = re.sub(dashes, " ", item)
        item = re.sub(multy_space, " ", item).strip()

        for line in item.split("\n"):
            pos = line.find(":")
            if pos != -1 and pos != len(line) - 1:
                character_name = line[:pos].strip()
                replica = line[pos + 2:].strip()
                raw_transcripts.append([speech_id, character_name, replica])
                speech_id += 1
            elif re.match(scene_delimiter_pattern, line.lower()) is not None:
                if raw_transcripts[-1][1] != scene_delimiter:
                    raw_transcripts.append([speech_id, scene_delimiter, scene_delimiter])
                    speech_id += 1
    return raw_transcripts


def download_data():
    with open("../data/scrubs/transcript_links.txt", "r") as links_file:
        links = [line.strip() for line in links_file]

    speech_id = 0
    with open("../data/scrubs/raw_transcripts.tsv", "w") as out_f:
        for link in tqdm(links):
            transcripts = get_raw_transcripts(link, speech_id)
            speech_id += len(transcripts)
            for item in transcripts:
                out_f.write("{}\t{}\t{}\n".format(*item))


if __name__ == "__main__":
    download_data()