import numpy as np
import re

# ssh yuj49@h2p.crc.pitt.edu

"""
Reformat file
Original format:
      PMID | t | Title text
      PMID | a | Abstract text
      PMID TAB StartIndex TAB EndIndex TAB MentionTextSegment TAB SemanticTypeID TAB EntityID
      
New format:
      Title1 text
      Abstract1 text 
      Selegiline	B-Chemical
      induced	O
      postural	B-Disease
      hypotension	I-Disease
      
      Title2 text
      Abstract2 text
      
"""


def produce_source_1(filename, dev, test, train):
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
  dev_raw = open("dev_1.raw", "w")
  test_raw = open("test_1.raw", "w")
  train_raw = open("train_1.raw", "w")
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


def turn_BOI(filename):
  # test_raw = open("test_1.raw", "w")
  # train_raw = open("train_1.raw", "w")
  train_nodes = []
  train="/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/raw/corpus_pubtator_pmids_dev.txt"
  with open(train, "r") as ftrain:
    for i in ftrain.readlines():
      train_nodes.append(i.split("\n")[0])
  final_text_list = {}
  with open(filename, 'r') as file:
    text = []
    for i in file.readlines():
      # make sure not the blank line.
      if i == "\n":
        text = []
        continue
      else:
        id = i[0:8]
        if id in train_nodes:
          if "|a|" in i:
            text.append(i.split("|a|")[1].split("\n")[0])
          elif "|t|" in i:
            text.append(i.split("|t|")[1].split("\n")[0])
          else: continue
        else:continue
        # get the full text
      if(len(text)==2):
        final_text = text[0] + " " + text[1]
        final_text_list[id] =final_text

  print("dict is",final_text_list)

  for final_id,final_text in final_text_list.items():
    print(final_id,final_text)
    write_text(filename, final_id,final_text)


def write_text(filename, final_id, final_text):
  train = open("dev.raw", "a")
  lis_ = final_text.split(" ")
  # patten_node = re.compile(r",|.|!|(|)|<|>|{|}|-|\|")

  # get all pairs with begin and end node
  # pairs like [[0,8],
  #             [12, 14]...]
  pairs = []
  with open(filename, 'r') as file:
    for i in file.readlines():
      if i[0:8] == final_id:
        print(final_id)
        pattern = re.compile(r"\|a\||\|t\|")
        if pattern.findall(i) == [] and i!="\n":
          all_item = i.split("\t")
          begin = int(all_item[1])
          end = int(all_item[2])
          pairs.append([begin, end])
      else:continue

  # get all
  count = 0
  for item in lis_:
    flag = False
    temp = re.split("(\W+)", item)

    # have ,.()[]{}... in item
    if len(temp) > 1 and"." not in temp:
      while "" in temp:
        temp.remove("")
      train.write(item + "\t" + "O" + "\n")
      count+=len(item)
      flag=True
      # for less_item in temp:
      #   # flag = Add_BI(count, less_item, pairs, flag)
      #   # -
      #   for pair in pairs:
      #     # when there is a single word as entity
      #     length = len(less_item)
      #     if count <pair[0] and count+length <pair[0]:
      #       flag=False
      #
      #       break
      #     if count == pair[0] and count + length == pair[1]:
      #       test_.write(less_item + "\t" + "B" + "\n")
      #       print("count", count)
      #       print("less_item", less_item)
      #       flag = True
      #       # count +=1
      #       count += len(less_item)
      #       break
      #     # when there is a multiple words as entity and this is the first
      #     elif count == pair[0] and count < pair[1] and length < pair[1] - pair[0]:
      #       test_.write(less_item + "\t" + "B" + "\n")
      #       flag = True
      #       # count += 1
      #       count += len(less_item)
      #       break
      #     # when there is a multiple words as entity and this is the rest
      #     elif count >= pair[0] and count < pair[1] and length < pair[1] - pair[0]:
      #       test_.write(less_item + "\t" + "I" + "\n")
      #       flag = True
      #       # count += 1
      #       count += len(less_item)
      #       break
      #     # this case for (CF) that ( index is less than pair[0]
      #     elif count < pair[0] and length + count <= pair[0]:
      #       count += length
      #       test_.write(less_item + "\t" + "O" + "\n")
      #       break
      #     elif count == pair[1]:
      #       test_.write(less_item + "\t" + "O" + "\n")
      #       break
      #
      #     else:
      #       continue
      count += 1
    else:
      # flag = Add_BI(count, item, pairs, flag)
      for pair in pairs:
        # when there is a single word as entity
        length = len(item)
        if count == pair[0] and count + length == pair[1]:
          train.write(item + "\t" + "B" + "\n")

          flag = True
          count += 1
          count += len(item)
          break

        # when there is a multiple words as entity
        elif count == pair[0] and count < pair[1] and length < pair[1] - pair[0]:
          train.write(item + "\t" + "B" + "\n")
          flag = True
          count += 1
          count += len(item)
          break

        elif count > pair[0] and count < pair[1] and length < pair[1] - pair[0]:
          train.write(item + "\t" + "I" + "\n")
          flag = True
          count += 1
          count += len(item)
          break
        elif count < pair[0]:
          break
        else:
          continue

    # if flag means before is entiy
    if flag:
      continue
    else:
      # add O
      temp = re.split("(\W+)", item)
      count += len(item)

      if len(temp) > 1:
        for i in temp:
          if i != "" and i != ".":
            train.write(i + "\t" + "O" + "\n")
          elif i == ".":
            train.write(i + "\t" + "O" + "\n")
            train.write("\r\n")
      else:
        train.write(item + "\t" + "O" + "\n")

    count += 1


if __name__ == "__main__":
  # filename = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/corpus_pubtator.txt"
  # dev = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/corpus_pubtator_pmids_dev.txt"
  # test = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/corpus_pubtator_pmids_test.txt"
  # train = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/corpus_pubtator_pmids_trng.txt"

  train = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/raw/corpus_pubtator.txt"

  test_for_this = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/raw/test.txt"
  # produce_source_1(filename, dev, test, train)



  turn_BOI(train)
