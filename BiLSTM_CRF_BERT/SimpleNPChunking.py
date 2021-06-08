import spacy


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


def get_doc_1(filename):
  test_node = []
  test_list = {}

def use_chunking(test_doc, test_list):
  P = 0.0
  R = 0.0
  TP = 0
  FN = 0
  FP = 0

  nlp = spacy.load("en_core_web_sm")
  for key, value in test_doc.item():
    doc = nlp(value)
    test = []
    for chunk in doc.noun_chunks:
      test.append(str(chunk))
    for i in test:
      if i in test_list[key]:
        TP+=1


  # doc = nlp("Autonomous cars shift insurance liability toward manufacturers")
  # for chunk in doc.noun_chunks:
  #     print(chunk.text, chunk.root.text, chunk.root.dep_,
  #             chunk.root.head.text)

if __name__ == "__main__":
  filename = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/raw/corpus_pubtator.txt"
  get_doc(filename)
