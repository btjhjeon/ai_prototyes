import fire
import csv
import tqdm
import math
from rouge import Rouge

from konlpy.tag import Mecab
from nltk.translate.bleu_score import sentence_bleu


def evaluate(
    data_path:str
):
    with open(data_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]

    mecab = Mecab()
    goldens = [row['golden'] for row in data]
    goldens = [mecab.morphs(g) for g in goldens]
    idf_dict = get_idf_dict(goldens)
    rouge = Rouge()

    score_pf1 = 0
    score_pcover = 0
    score_bleu = 0  # BLEU-1
    score_rouge = 0 # ROUGE-1
    num_data = len(data)
    for i, row in tqdm.tqdm(enumerate(data), total=len(data)):
        history = [*row['personas'].split("\n"), row['golden']]
        history = [mecab.morphs(r) for r in history]
        result = mecab.morphs(row['response'])
        golden = goldens[i]

        score_pcover += calculate_pcover(result, golden, idf_dict)
        score_pf1 += calculate_pf1(result, history)
        score_bleu += sentence_bleu([' '.join(h) for h in history], ' '.join(result), weights=(1, 0, 0, 0))
        score_rouge += rouge.get_scores(' '.join(result), ' '.join(golden))[0]["rouge-1"]['r']
    score_pcover /= num_data
    score_pf1 /= num_data
    score_bleu /= num_data
    score_rouge /= num_data
    print(f"P-Cover: {score_pcover:.4f}")
    print(f"P-F1:\t {score_pf1:.4f}")
    print(f"BLEU-1:\t {score_bleu:.4f}")
    print(f"ROUGE-1: {score_rouge:.4f}")


def get_idf_dict(bloblist):
    # returns the number of documents containing word
    def n_containing(word, bloblist):
        return sum(1 for blob in bloblist if word in blob)

    # computes "inverse document frequency" which measures how common a word is among all documents in bloblist.
    def idf(word, bloblist):
        return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))
        # ratio of the total number of documents to the number of documents containing word
    
    words = set(sum(bloblist, []))
    idf_dict = {}
    for word in words:
        idf_dict[word] = idf(word, bloblist)
    return idf_dict


def calculate_pf1(result, history):
    h_all = []
    for h in history:
        h_all += h

    # h_set = set(h_all) - set(stop_words_set)
    # r_set = set(result) - set(stop_words_set)
    h_set = set(h_all)
    r_set = set(result)

    if len(h_set) == 0 or len(r_set) == 0:
        p, r = 0, 0
    else:
        p = len(h_set & r_set) / len(r_set)
        r = len(h_set & r_set) / len(h_set)

    # print(p,r)
    if p == r == 0:
        return 0
    return (2 * p * r) / (p + r)


def calculate_pcover(result, reference, idf_dict):
    if len(result) == 0:
        return 0

    c = 0
    has_c = {}
    for w in result:
        if w in reference and w not in has_c:
            # c += idf_dict[w]
            if w in idf_dict:
                c += idf_dict[w]
                has_c[w] = 1
    return c / len(result)


if __name__ == "__main__":
    fire.Fire(evaluate)
