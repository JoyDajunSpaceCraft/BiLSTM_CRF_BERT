"""
not consider I before B
"""


def get_test_result(filename):
    original_list = []
    res_list = []
    with open(filename, "r") as f:
        chunk_original = []
        start_flag_original = False
        chunk_res = []
        start_flag_res = False
        for item in f.readlines():
            if item != "\n":
                new_item = item.split(" ")
                if item.split(" ")[1] == "B":
                    chunk_original.append(item.split(" ")[0])
                    start_flag_original = True
                elif item.split(" ")[1] == "I":
                    chunk_original.append(item.split(" ")[0])
                elif item.split(" ")[1] == "O" and start_flag_original == True:
                    original_list.append(chunk_original)
                    chunk_original = []
                    start_flag_original = False

                if item.split(" ")[2] == "B\n":
                    chunk_res.append(item.split(" ")[0])
                    start_flag_res = True
                elif item.split(" ")[2] == "I\n":
                    chunk_res.append(item.split(" ")[0])
                elif item.split(" ")[2] == "O\n" and start_flag_res == True:
                    res_list.append(chunk_res)
                    chunk_res = []
                    start_flag_res = False
                else:
                    continue
            else:
                continue

    print("res:", res_list)
    print("original", original_list)
    with open("res_list.txt", "a") as result:

        for res, ori in zip(res_list, original_list):
            if len(res) > 1:
                for item in res[:-1]:
                    result.write(item + " ")
                result.write(res[-1] + "\n")
            else:
                result.write(res[0] + "\n")

            if len(ori) > 1:
                for ori_ in ori[:-1]:
                    result.write(ori_ + " ")
                result.write(ori[-1] + "\n")
            else:
                result.write(ori[0])
            result.write("\n")

    # with open("get true","w")


def get_sen(filepath1, filepath2):
    sen1 = []
    sen2 = []
    chunk1 = []
    chunks1 = []
    chunk2 = []
    chunks2 = []
    with open(filepath1, "r") as test:
        start_flag = False
        count = 0
        for item in test.readlines():

            if item != "\n":
                if item.split("\t")[1] == "B\n":
                    count += 1
                if item.split("\t")[1] == "B\n":
                    chunk1.append(item.split("\t")[0])
                    start_flag = True
                elif item.split("\t")[1] == "I\n":
                    chunk1.append(item.split("\t")[0])
                elif item.split("\t")[1] == "O\n" and start_flag == True:
                    chunks1.append(chunk1)
                    chunk1 = []
                    start_flag = False

            else:
                sen1.append(chunks1)
                chunks1 = []

    with open(filepath2, "r") as test:
        start_flag = False
        for item in test.readlines():
            if item != "\n":
                if item.split(" ")[2] == "B\n":
                    chunk2.append(item.split(" ")[0])
                    start_flag = True
                elif item.split(" ")[2] == "I\n":
                    chunk2.append(item.split(" ")[0])
                elif item.split(" ")[2] == "O\n" and start_flag == True:
                    chunks2.append(chunk2)
                    chunk2 = []
                    start_flag = False

            else:
                sen2.append(chunks2)
                chunks2 = []
    print(len(sen1), len(sen2))
    print("sen1", sen1)
    print("sen2", sen2)
    CO = 0
    POA = 0
    PO = 0
    COA = 0
    for index, value in enumerate(sen1):
        # for idx, i in enumerate(value):
        #     entity = ""
        #     if len(i)>1:
        #         for j in i[:-1]:
        #             entity+=j+" "
        #         entity+=i[-1]
        #     else: entity = i[0]
        #     value[idx] = entity
        for i in value:
            # i is ['analyses', 'tissue', 'microarray'] an entity
            if i not in sen2[index]:
                for j in i:
                    # j is analyses...
                    # sen2[index] is ['analyses', 'tissue', 'microarray'],[...]
                    for n in sen2[index]:
                        # n is ['analyses', 'tissue', 'microarray']
                        if j in n and len(i) < len(n):
                            COA += 1
                        elif j in n and len(i) > len(n):
                            PO += 1
                        elif j in n and len(i) == len(n):
                            POA += 1

        # for n in sen2[index]:
        #
        #     for m in n:
        #
        #     if i[0] in sen2[index] and i not in sen2[index]:
        #         POA =
        # print(value[idx])
        # if i in sen2[index]:
        #     CO+=1
    print("total", count)
    print(COA, PO, POA)


def get_parse(testfile):
    res_list = []
    chunk_res = []
    start_flag_res = False
    with open(testfile, "r") as test:
        for item in test.readlines():
            if item != "\n":
                # include in B I I O / B I I B /
                if item.split(" ")[2] == "B\n" and start_flag_res == False:
                    chunk_res.append(item.split(" ")[0])
                    start_flag_res = True
                elif item.split(" ")[2] == "I\n":
                    # print("we get I")
                    chunk_res.append(item.split(" ")[0])
                elif item.split(" ")[2] == "O\n" and start_flag_res == True:
                    res_list.append(chunk_res)
                    chunk_res = []
                    start_flag_res = False

                # that means the before one is not O
                elif item.split(" ")[2] == "B\n" and start_flag_res == True:
                    print("this chunk res is", chunk_res)
                    res_list.append(chunk_res)
                    chunk_res = []
                    chunk_res.append(item.split(" ")[0])
                    start_flag_res = True
                else:
                    continue
            else:
                continue
        # need to add last one
        res_list.append(chunk_res)
    if res_list[-1] == []:
        res_list = res_list[:-1]
    return res_list


def count_spacytype(pos_tagging):
    count_verb = 0
    count_right_verb = 0

    count_noun = 0
    count_right_noun = 0

    count_adj = 0
    count_right_adj = 0

    count_adv = 0
    count_right_adv = 0

    with open(pos_tagging, "r") as verb:
        for i in verb.readlines():
            if i != "\n":
                if i.split(" ")[1] == "VERB":
                    count_verb += 1
                elif i.split(" ")[1] == "NOUN":
                    count_noun += 1
                elif i.split(" ")[1] == "ADJ":
                    count_adj += 1
                elif i.split(" ")[1] == "ADV":
                    count_adv += 1
                # print(len(i.split(" ")))
                # print(i.split(" "))
                if len(i.split(" ")) == 4:
                    if i.split(" ")[1] == "VERB" and i.split(" ")[2] == i.split(" ")[3].split("\n")[0]:
                        count_right_verb += 1
                        # print(count_right_verb)
                    if i.split(" ")[1] == "NOUN" and i.split(" ")[2] == i.split(" ")[3].split("\n")[0]:
                        count_right_noun += 1
                    if i.split(" ")[1] == "ADJ" and i.split(" ")[2] == i.split(" ")[3].split("\n")[0]:
                        count_right_adj += 1
                    if i.split(" ")[1] == "ADV" and i.split(" ")[2] == i.split(" ")[3].split("\n")[0]:
                        count_right_adv += 1

            else:
                continue
    print(count_verb)
    print(count_noun)
    print(count_adj)
    print(count_adv)
    print(count_right_verb / count_verb)
    print(count_right_noun / count_noun)
    print(count_right_adj / count_adj)
    print(count_right_adv / count_adv)
    print("*" * 8)
    print(1 - count_right_verb / count_verb)
    print(1 - count_right_noun / count_noun)
    print(1 - count_right_adj / count_adj)
    print(1 - count_right_adv / count_adv)


def get_1gram(testfile):
    count1gram = 0
    res_list = []
    chunk_res = []
    start_flag_res = False
    right_1gram = 0
    overlap_1gram = 0
    wrong_1gram = 0
    chunk_t = []
    t_list = []

    with open(testfile, "r") as test:
        index = 0
        idx = []
        true_data = []
        test_data = []
        for item in test.readlines():
            if item != "\n":
                true_data.append(item.split(" ")[1])
                test_data.append(item.split(" ")[2].split("\n")[0])
                if item.split(" ")[2] == "B\n" and start_flag_res == False:
                    chunk_res.append(item.split(" ")[0])
                    start_flag_res = True
                elif item.split(" ")[2] == "I\n":
                    chunk_res.append(item.split(" ")[0])
                elif item.split(" ")[2] == "O\n" and start_flag_res == True:
                    res_list.append(chunk_res)
                    if (len(chunk_res) == 1):
                        count1gram += 1
                        idx.append(index - 1)
                    chunk_res = []
                    start_flag_res = False

                # that means the before one is not O
                elif item.split(" ")[2] == "B\n" and start_flag_res == True:
                    print("this chunk res is", chunk_res)
                    res_list.append(chunk_res)
                    if (len(chunk_res) == 1):
                        count1gram += 1
                        idx.append(index - 1)
                    chunk_res = []
                    chunk_res.append(item.split(" ")[0])
                    start_flag_res = True
                index += 1
            else:
                continue
        print("total 1gram in true", count1gram)
        for i in idx:
            if true_data[i] == test_data[i] and test_data[i + 1] in ["B", "O"]:
                right_1gram += 1
            elif true_data[i] == test_data[i] and test_data[i + 1] == "I":
                overlap_1gram += 1
            else:
                wrong_1gram += 1
        print("right is", right_1gram)
        print("overlap is", overlap_1gram)
        print("wrong is", wrong_1gram)

        # need to add last one
    #     res_list.append(chunk_res)
    # if res_list[-1] == []:
    #     res_list = res_list[:-1]


def get_2gram(testfile):
    count2gram = 0
    res_list = []
    chunk_res = []
    start_flag_res = False
    right_2gram = 0
    overlap_2gram = 0
    wrong_2gram = 0
    wrong_overlap = 0
    right_less2gram = 0
    with open(testfile, "r") as test:
        index = 0
        idx = []
        true_data = []
        test_data = []
        for item in test.readlines():
            if item != "\n":
                true_data.append(item.split(" ")[1])
                test_data.append(item.split(" ")[2].split("\n")[0])
                if item.split(" ")[2] == "B\n" and start_flag_res == False:
                    chunk_res.append(item.split(" ")[0])
                    start_flag_res = True
                elif item.split(" ")[2] == "I\n":
                    chunk_res.append(item.split(" ")[0])
                elif item.split(" ")[2] == "O\n" and start_flag_res == True:
                    res_list.append(chunk_res)
                    if (len(chunk_res) == 2):
                        count2gram += 1
                        idx.append(index - 2)
                        print(test_data[index - 2])

                    chunk_res = []
                    start_flag_res = False

                # that means the before one is not O
                elif item.split(" ")[2] == "B\n" and start_flag_res == True:
                    # print("this chunk res is", chunk_res)
                    res_list.append(chunk_res)
                    if (len(chunk_res) == 2):
                        count2gram += 1
                        idx.append(index - 2)
                    chunk_res = []
                    chunk_res.append(item.split(" ")[0])
                    start_flag_res = True
                index += 1
            else:
                continue
        print("total 2gram in true", count2gram)
        for i in idx:
            if true_data[i] == test_data[i] and true_data[i + 1] == "I" and true_data[i + 2] in ["B", "O"]:
                right_2gram += 1
            elif true_data[i] == test_data[i] and true_data[i + 2] == "I" and true_data[i + 1] == "I":
                overlap_2gram += 1
            elif (true_data[i] == "O" and true_data[i + 1] == "B") or (true_data[i] == "B" and true_data[i + 1] == "O"):
                right_less2gram += 1
            elif true_data[i] in ["O", "I"] or true_data[i + 1] in ["O", "B"]:
                wrong_overlap += 1

            # print(test_data[i])
        print("right is", right_2gram)
        print("overlap is", overlap_2gram)
        print("right_less2gram is", right_less2gram)
        print("wrong_overlap is", wrong_overlap)


def get_3gram(testfile):
    count3gram = 0
    res_list = []
    chunk_res = []
    start_flag_res = False
    right_3gram = 0
    overlap_3gram = 0
    wrong_3gram = 0
    wrong_overlap = 0
    right_less3gram = 0
    with open(testfile, "r") as test:
        index = 0
        idx = []
        true_data = []
        test_data = []
        for item in test.readlines():
            if item != "\n":
                true_data.append(item.split(" ")[1])
                test_data.append(item.split(" ")[2].split("\n")[0])
                if item.split(" ")[2] == "B\n" and start_flag_res == False:
                    chunk_res.append(item.split(" ")[0])
                    start_flag_res = True
                elif item.split(" ")[2] == "I\n":
                    chunk_res.append(item.split(" ")[0])
                elif item.split(" ")[2] == "O\n" and start_flag_res == True:
                    res_list.append(chunk_res)
                    if (len(chunk_res) == 3):
                        count3gram += 1
                        idx.append(index - 3)
                        print(test_data[index - 3])

                    chunk_res = []
                    start_flag_res = False

                # that means the before one is not O
                elif item.split(" ")[2] == "B\n" and start_flag_res == True:
                    # print("this chunk res is", chunk_res)
                    res_list.append(chunk_res)
                    if (len(chunk_res) == 3):
                        count3gram += 1
                        idx.append(index - 3)
                    chunk_res = []
                    chunk_res.append(item.split(" ")[0])
                    start_flag_res = True
                index += 1
            else:
                continue
        print("total 2gram in true", count3gram)
        for i in idx:
            temp = true_data[i]+true_data[i+1]+true_data[i+2]
            if true_data[i] == test_data[i] and true_data[i + 1] == "I" and true_data[i + 2] == "I"and true_data[i + 3] in ["B", "O"]:
                right_3gram += 1
            elif true_data[i] == test_data[i] and true_data[i + 3] == "I" and true_data[i + 2] == "I" and true_data[i + 1] == "I":
                overlap_3gram += 1
            elif temp in ["BIO","OBI"]:
                right_less3gram += 1
            elif true_data[i] in ["O", "I"] or true_data[i + 1] in ["O", "B"] or true_data[i + 2] in ["O", "B"]:
                wrong_overlap += 1

            # print(test_data[i])
        print("right is", right_3gram)
        print("overlap is", overlap_3gram)
        print("right_less2gram is", right_less3gram)
        print("wrong_overlap is", wrong_overlap)


if __name__ == "__main__":
    # filename = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/test_get_parse.txt"
    # get_parse(filename)
    true_test = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/all_token_labels_.txt"
    # get_1gram(true_test)
    # get_2gram(true_test)
    get_3gram(true_test)

    # # get_test_result(filename)
    # filepath = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/formal/test.txt"

    # get_sen(filepath,filename)
    # pos_tagging = "/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/pos_tagging4.txt"
    # count_spacytype(pos_tagging)
