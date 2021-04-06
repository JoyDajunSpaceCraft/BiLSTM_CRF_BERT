import numpy as np

"""
Reformat file
Original format:
      PMID | t | Title text
      PMID | a | Abstract text
      PMID TAB StartIndex TAB EndIndex TAB MentionTextSegment TAB SemanticTypeID TAB EntityID
New format:
      Title1 text
      Abstract1 text 
      
      Title2 text
      Abstract2 text
      
"""


def produce_source(filename, dev, test, train):
  abstractList = []
  titleList = []
  labelList = []

  dev_nodes = []
  test_nodes = []
  train_nodes = []
  with open(dev, "r") as fdev:
    for i in fdev.readlines():
      dev_nodes.append(i.split("\n")[0])
  with open(test, "r") as ftest:
    for i in ftest.readlines():
      test_nodes.append(i.split("\n")[0])
  with open(train, "r") as ftrain:
    for i in ftrain.readlines():
      train_nodes.append(i.split("\n")[0])

  # file = open("source.txt", "w")
  dev_raw = open("dev.raw", "w")
  test_raw = open("test.raw","w")
  train_raw = open("train.raw","w")
  with open(filename, "r")as f:
    for i in f.readlines():
      if "|t|" in i:
        node_id = i.split("|t|")[0]
        # print(node_id)
        title = i.split("|t|")[1]
        if node_id in dev_nodes:
          dev_raw.write(title)
        elif node_id in test_nodes:
          test_raw.write(title)
        else:
          train_raw.write(title)

      # titleList.append(i.split("|t|")[1])

      if "|a|" in i:
        node_id = i.split("|a|")[0]
        abstract = i.split("|a|")[1]
        if node_id in dev_nodes:
          dev_raw.write(abstract)
          dev_raw.write("\n")
        elif node_id in test_nodes:
          test_raw.write(abstract)
          test_raw.write("\n")
        else:
          train_raw.write(abstract)
          train_raw.write("\n")
        # abstractList.append(i.split("|a|")[1])



      else:
        labelList.append(i)


if __name__ == "__main__":
  filename = "/Users/apple/PycharmProjects/fairseq/BiLSTM_CRF_BERT/data/MedMentions/corpus_pubtator.txt"
  dev = "/Users/apple/PycharmProjects/fairseq/BiLSTM_CRF_BERT/data/MedMentions/corpus_pubtator_pmids_dev.txt"
  test = "/Users/apple/PycharmProjects/fairseq/BiLSTM_CRF_BERT/data/MedMentions/corpus_pubtator_pmids_test.txt"
  train = "/Users/apple/PycharmProjects/fairseq/BiLSTM_CRF_BERT/data/MedMentions/corpus_pubtator_pmids_trng.txt"

  produce_source(filename, dev, test, train)
