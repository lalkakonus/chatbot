import os

def update_formated_data(input_path, output_path):
    with open(input_path, encoding='utf8') as in_f:
        with open(output_path, "a", encoding='utf8') as out_f:
            last_num = 0
            last_response = None
            for line in in_f.readlines():
                splited = line.split('\t')
                num = int(splited[0].split()[0])
                replica = ' '.join(splited[0].split()[1:])
                response = splited[1]
                if last_num >= num:
                    last_response = None
                if last_response is not None:
                    out_f.write("{}\t{}\n".format(last_response, replica))
                out_f.write("{}\t{}\n".format(replica, response))

                last_num = num
                last_response = response


if __name__ == "__main__":
    corpus_path = "../../data/personachat"
    datafile = os.path.join(corpus_path, "formatted_movie_lines.txt")

    for filename in os.listdir(corpus_path):
        update_formated_data(os.path.join(corpus_path, filename), datafile)

