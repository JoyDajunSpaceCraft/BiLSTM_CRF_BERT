# this function is how to caculate the whole B in the test file
def countB():
    count = 0
    filename = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/formal/test.txt"
    with open(filename, "r")as f:
        for i in f.readlines():
            if i != "\n":
                print(i)
                if len(i.split("\t")) == 2 and i.split("\t")[1] == "B\n":
                    count += 1
                else:
                    print(i)
    print(count)


def get_all_percent():
    filename = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/pos_tagging_7_11.txt"
    countNoun = 0
    countVerb = 0
    countAdj = 0
    countAdv = 0

    pos_tag = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/pos_tagging.txt"

    countTNoun = 0
    countTVerb = 0
    countTAdj = 0
    countTAdv = 0

    with open(pos_tag, "r") as pos:
        for i in pos.readlines():
            if i != "\n" or "SPACE 103" not in i and len(i.split(" ")) == 3:
                if i.split(" ")[1] == "NOUN":
                    countTNoun += 1
                elif i.split(" ")[1] == "VERB":
                    countTVerb += 1
                elif i.split(" ")[1] == "ADJ":
                    countTAdj += 1
                elif i.split(" ")[1] == "ADV":
                    countTAdv += 1

            else:
                continue

    with open(filename, "r") as f:
        for i in f.readlines():
            if i != "\n":
                if i.split(" ")[1] == "NOUN" and (i.split(" ")[2] == "B" or i.split(" ")[2] == "I"):
                    countNoun += 1
                elif i.split(" ")[1] == "VERB" and (i.split(" ")[2] == "B" or i.split(" ")[2] == "I"):
                    countVerb += 1

                elif i.split(" ")[1] == "ADV" and (i.split(" ")[2] == "B" or i.split(" ")[2] == "I"):
                    countAdv += 1
                elif i.split(" ")[1] == "ADJ" and (i.split(" ")[2] == "B" or i.split(" ")[2] == "I"):
                    countAdj += 1
    print("percentage of noun", countNoun / countTNoun)
    print("percentage of verb", countVerb / countTVerb)
    print("percentage of adj", countAdj / countTAdj)
    print("percentage of adv", countAdv / countTAdv)

    print("percentage of noun", 1 - countNoun / countTNoun)
    print("percentage of verb", 1 - countVerb / countTVerb)
    print("percentage of adj", 1 - countAdj / countTAdj)
    print("percentage of adv", 1 - countAdv / countTAdv)

    print("count of true noun", countTNoun)
    print("count of true verb", countTVerb)
    print("count of true adj", countTAdj)
    print("count of true adv", countTAdv)

    print("count of noun", countNoun)
    print("count of verb", countVerb)
    print("count of adi", countAdj)
    print("count of adv", countAdv)


def get_gram_total_num():
    filename = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/formal/test.txt"
    res_list = []
    count1 = 0
    
    start_flag_res = False
    chunk_res = []
    with open(filename, "r") as f:
        for item in f.readlines():
            if item != "\n" and len(item.split("\t")) == 2:
                if item.split("\t")[1] == "B\n" and start_flag_res == False:
                    chunk_res.append(item.split("\t")[0])
                    start_flag_res = True
                elif item.split("\t")[1] == "I\n":
                    chunk_res.append(item.split("\t")[0])
                elif item.split("\t")[1] == "O\n" and start_flag_res == True:
                    # print("this chunk res is", chunk_res)
                    res_list.append(chunk_res)
                    if len(chunk_res) == 3:
                        count1 += 1
                    chunk_res = []
                    start_flag_res = False

                    # that means the before one is not O
                elif item.split("\t")[1] == "B\n" and start_flag_res == True:
                    res_list.append(chunk_res)
                    if len(chunk_res) == 3:
                        count1 += 1
                    chunk_res = []
                    chunk_res.append(item.split("\t")[0])
                    start_flag_res = True
                else:
                    continue
            else:
                continue
    print(count1)






if __name__ == "__main__":
    get_all_percent()
    # get_gram_total_num()
    # countB()
    # pass

