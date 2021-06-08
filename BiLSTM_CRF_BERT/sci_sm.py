import spacy
import scispacy
import os

def get_doc(filename):
    test_node = []
    test_list = {}

    with open("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/raw/corpus_pubtator_pmids_test.txt",
              "r") as test:
        for i in test.readlines():
            test_node.append(i.split("\n")[0])
            test_list[i.split("\n")[0]] = []

    test_docs = {}
    with open(filename, "r") as fdev:
        for i in fdev.readlines():
            # test_docs.append(i.split("\n")[0])

            if "|t|" in i and i.split("|t|")[0] in test_node:
                node_id = i.split("|t|")[0]
                if "|t|" in i:
                    title = i.split("|t|")[1].split("\n")[0]
                    test_docs[node_id] = title

            elif "|a|" in i and i.split("|a|")[0] in test_node:
                node_id = i.split("|a|")[0]
                if "|a|" in i:
                    ab = i.split("|a|")[1].split("\n")[0]
                    test_docs[node_id] += ab
            elif i != "\n" and i.split("\t")[0] in test_node:
                node_id = i.split("\t")[0]
                test_list[node_id].append(i.split("\t")[3])
            else:
                continue
    print(test_docs)
    print(test_list)
    return test_docs, test_list


def use_sci_sm(test_docs, test_list):
    P = 0.0
    R = 0.0
    TP = 0
    FN = 0
    FP = 0

    nlp = spacy.load("en_core_sci_sm")

    for key, value in test_docs.items():
        doc = nlp(value)
        test = []
        for sm_ent in doc.ents:
            test.append(str(sm_ent))
        FP += len(test)
        for i in test:
            if i in test_list[key]:
                TP += 1

    for i in test_list.values():
        FN += len(i)
    P = TP / FP
    R = TP / FN
    F = 2 * P * R / (P + R)
    print(P, R, F)


def use_sci_md(test_docs, test_list):
    P = 0.0
    R = 0.0
    TP = 0
    FN = 0
    FP = 0

    nlp = spacy.load("en_core_sci_md")
    for key, value in test_docs.items():
        doc = nlp(value)
        test = []
        for md_ent in doc.ents:
            test.append(str(md_ent))
        FP += len(test)
        for i in test:
            if i in test_list[key]:
                TP += 1

    for i in test_list.values():
        FN += len(i)
    P = TP / FP
    R = TP / FN
    F = 2 * P * R / (P + R)

    print(P, R, F)


def use_sci_scibert(test_docs, test_list):
    P = 0.0
    R = 0.0
    TP = 0
    FN = 0
    FP = 0

    nlp = spacy.load("en_core_sci_scibert")
    for key, value in test_docs.items():
        doc = nlp(value)
        test = []
        for scibert_ent in doc.ents:
            test.append(str(scibert_ent))
        FP += len(test)
        for i in test:
            if i in test_list[key]:
                TP += 1

    for i in test_list.values():
        FN += len(i)
    P = TP / FP
    R = TP / FN
    F = 2 * P * R / (P + R)

    print(P, R, F)

def test_file(filename):
    nlp = spacy.load("en_core_sci_sm")
    doc = []
    # with open("test.txt", "a") as test_file:
    for file in os.listdir(filename):
        filepath = os.path.join(filename, file)
        with open(filepath, "r") as f:
            for item in f.readlines():
                doc.append(item)
    test_result = []
    for i in doc:
        value = nlp(i)
        for chunk in value.noun_chunks:
            test_result.append(str(chunk))
    with open("sci_sm.txt", "w") as f:
        for i in test_result:
            f.write(i + "\r\n")

if __name__ == "__main__":
    # filename = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/raw/corpus_pubtator.txt"
    filename= "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/test_example"
    test_file(filename)
    # test_docs, test_list = get_doc(filename)
    # use_sci_md(test_docs, test_list)
    # use_sci_sm(test_docs, test_list)
    # use_sci_scibert(test_docs, test_list)
