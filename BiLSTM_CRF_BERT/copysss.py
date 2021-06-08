import os
import re

store_list = set()
def cmd_data(test_docs, test_list):
    P = 0.0
    R = 0.0
    TP = 0
    FN = 0
    FP = 0
    for key, value in test_docs.items():
        store_list = set()
        if "(" or ")" in value:
            value = value.replace("("," ")
            value = value.replace(")"," ")
        info = os.popen(
            " echo {} | ./bin/metamap -I ".format(value)).readlines()

        for line in info:
            if "Disease or Syndrome" in line or "Functional Concept" in line  or "Therapeutic or Preventive Procedure" in line and re.match(r"(.*)\s*C+\d+",line, re.M|re.I)!=None:
                item = line.split(":")[1].split(" ")[0]
                store_list.add(item)
        FP += len(store_list)
        for i in store_list:
            if i in test_list[key]:
                TP += 1

    for i in test_list.values():
        FN += len(i)
    P = TP / FP
    R = TP / FN
    F = 2 * P * R / (P + R)
    print(P, R, F)


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

    return test_docs, test_list


if __name__ == "__main__":
    filename = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/raw/corpus_pubtator.txt"
    test_docs, test_list = get_doc(filename)
    cmd_data(test_docs, test_list)

