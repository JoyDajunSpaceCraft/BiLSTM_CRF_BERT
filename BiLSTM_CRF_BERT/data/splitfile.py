# import os
#
# with open("test.txt", "a") as test_file:
#     for file in os.listdir("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/test_example"):
#         filepath = os.path.join("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/test_example", file)
#         with open(filepath, "r") as f:
#             for item in f.readlines():
#                 s_flag = False
#                 if "." in item:
#                     s_flag = True
#                 raw = item.split(" ")
#                 for j in raw:
#                     j = j.replace(",", "")
#                     j = j.replace(":", "")
#                     j = j.replace(".", "")
#                     j = j.replace("/", "")
#                     j = j.replace('\"', "")
#                     j = j.replace("(", "")
#                     j = j.replace(")", "")
#                     j = j.replace("\n", "")
#
#                     test_file.write(j + "\t" + "O" + "\r\n")
#
#                 if s_flag:
#                     test_file.write("\r\n")
import scispacy
# import en_core_web_sm
# import en_core_sci_sm

# nlp = spacy.load("en_core_web_sm")
import spacy
# nlp = en_core_sci_sm.load()


def get_pos_tagging():
    # test_nodes = []
    # docs = {}
    # with open("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/raw/corpus_pubtator_pmids_test.txt",
    #           "r") as ftest:
    #     for i in ftest.readlines():
    #         test_nodes.append(i.split("\n")[0])
    #         docs[i.split("\n")[0]] = ""
    # with open("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/raw/corpus_pubtator.txt",
    #           "r") as test_file:
    #     for i in test_file.readlines():
    #         if "|t|" in i:
    #             node_id = i.split("|t|")[0]
    #             # print(node_id)
    #             title = i.split("|t|")[1]
    #             if node_id in test_nodes:
    #                 docs[node_id] = title
    #             else:
    #                 continue
    #         if "|a|" in i:
    #             node_id = i.split("|a|")[0]
    #             # print(node_id)
    #             ab = i.split("|a|")[1]
    #             if node_id in test_nodes:
    #                 docs[node_id] += " " + ab
    #             else:
    #                 continue
    docs = {}
    with open("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/scibert_token_labels_.txt",
              "r") as test:
        # with open("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/test_get_parse.txt") as test:
        temp_doc = ""
        true_type = []
        test_type = []

        for i in test.readlines():
            if i != "\n":
                temp_doc += i.split(" ")[0] + " "
                true_type.append(i.split(" ")[1])
                test_type.append(i.split(" ")[2].split("\n")[0])
            else:
                docs[temp_doc] = [true_type, test_type]
                # docs.append("\n")
                temp_doc = ""
                true_type = []
                test_type = []
        docs[temp_doc] = [true_type, test_type]

    with open("pos_tagging4.txt", "w")as pos_tag:
        for key, value in docs.items():
            doc = nlp(key)

            for idx, token in enumerate(doc):
                pos_tag.write(str(token))
                pos_tag.write(" ")
                pos_tag.write(token.pos_)
                pos_tag.write(" ")
                keys = key.split(" ")
                # Nonylphenol diethoxylate inhibits apoptosis induced in PC12 cells Nonylphenol
                keys = keys[:-1]
                for index, v in enumerate(keys):
                    if str(token) == v and index <= idx:
                        pos_tag.write(value[0][index])
                        pos_tag.write(" ")
                        pos_tag.write(value[1][index])
                    else:
                        continue
                pos_tag.write("\n")
            pos_tag.write("\n")


def raw_test_transfer():
    docs = {}
    temp_doc = ""
    true_type = []
    # nlp = en_core_web_sm.load()
    nlp = spacy.load("en_core_sci_sm")
    res_list = []

    # exactly match
    chunk_right = 0
    # test result get more than expected
    overlap_chunk = 0
    # test result is not match
    right_less_chunk = 0

    start_flag_res = False
    chunk_res = []
    wrong_chunk = 0
    count_all = 0

    with open("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/formal/test.txt", "r") as test_file:
        # with open("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/test_get_parse.txt", "r") as test_file:
        for item in test_file.readlines():
            if item != "\n":
                temp_doc += item.split("\t")[0] + " "
                # true_type.append(item.split(" ")[1])

                if item.split("\t")[1] == "B\n" and start_flag_res == False:
                    chunk_res.append(item.split("\t")[0])
                    start_flag_res = True
                elif item.split("\t")[1] == "I\n":
                    chunk_res.append(item.split("\t")[0])
                elif item.split("\t")[1] == "O\n" and start_flag_res == True:
                    # print("this chunk res is", chunk_res)
                    res_list.append(chunk_res)
                    if len(chunk_res) == 3:
                        true_type.append(chunk_res[0] + " " + chunk_res[1]+" " + chunk_res[2])
                    chunk_res = []
                    start_flag_res = False

                # that means the before one is not O
                elif item.split("\t")[1] == "B\n" and start_flag_res == True:
                    res_list.append(chunk_res)
                    if len(chunk_res) == 3:
                        true_type.append(chunk_res[0] + " " + chunk_res[1]+" " + chunk_res[2])
                    chunk_res = []
                    chunk_res.append(item.split("\t")[0])
                    start_flag_res = True
                else:
                    continue
            else:
                docs[temp_doc] = true_type
                temp_doc = ""
                true_type = []
        # print("true type is", true_type)
        # docs[temp_doc] = true_type
    for key, value in docs.items():
        # print("key is", key)
        # print("value is", value)
        doc = nlp(key)
        print(doc.ents)
        # for chunk in doc.noun_chunks:
        for chunk in doc.ents:
            print(chunk)
            chunk = str(chunk)
            if len(chunk.split(" ")) == 3:
                count_all += 1
            if len(chunk.split(" ")) == 3 and chunk in value:
                chunk_right += 1
            elif len(chunk.split(" ")) > 3:
                for i in value:
                    if i.split(" ")[0] in chunk.split(" ") or i.split(" ")[1] in chunk.text.split(" ")or i.split(" ")[2] in chunk.text.split(" "):
                        overlap_chunk += 1
                        break
            elif len(chunk.split(" ")) == 3 and chunk not in value:
                for v in value:
                    if chunk.split(" ")[0] in v.split(" ") or chunk.split(" ")[1] in v.split(" ") or chunk.text.split(" ")[2] in v.split(" "):
                        wrong_chunk+=1
                        break
            if len(chunk.split(" ")) == 1 or len(chunk.split(" ")) == 2:
                for v in value:
                    if chunk in v.split(" "):
                        right_less_chunk+=1
                        break

    print(count_all)
    print(chunk_right)
    print(overlap_chunk)
    print(wrong_chunk)
    print(right_less_chunk)


if __name__ == "__main__":
    # get_pos_tagging()
    raw_test_transfer()

    # noun = []
    # verb = []
    # adj = []
    # adv = []
    # count_noun = 0
    # count_verb = 0
    # count_adj = 0
    # count_adv = 0
    #
    # with open("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/pos_tagging.txt", "r") as f:
    #     for i in f.readlines():
    #         if len(i.split(" "))==3 and i!="\n":
    #             if "NOUN" == i.split(" ")[1]:
    #                 noun.append(i)
    #                 count_noun += 1
    #             elif "VERB" == i.split(" ")[1]:
    #                 verb.append(i)
    #                 count_verb += 1
    #             elif "ADJ" == i.split(" ")[1]:
    #                 adj.append(i)
    #                 count_adj += 1
    #             elif "ADV" == i.split(" ")[1]:
    #                 adv.append(i)
    #                 count_adv += 1
    #             else:
    #                 continue
    #         else:
    #             continue
    # with open("noun.txt", "a")as nouns:
    #     for i in noun:
    #         nouns.write(i)
    # with open("verb.txt", "a")as verbs:
    #     for i in verb:
    #         verbs.write(i)
    # with open("adj.txt", "a")as adjs:
    #     for i in adj:
    #         adjs.write(i)
    # with open("adv.txt", "a")as advs:
    #     for i in adv:
    #         advs.write(i)
    #
    # print(count_noun,count_verb,count_adj,count_adv)

#     for file in os.listdir("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/test_example"):
#         filepath = os.path.join("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/test_example", file)
#         with open(filepath, "r") as f:
#             for item in f.readlines():
#               doc.append(item)
# test_result = []
# for i in doc:
#   value = nlp(i)
#   for chunk in value.noun_chunks:
#     test_result.append(str(chunk))
# with open("sci_sm_new.txt","w") as f:
#   for i in test_result:
#     f.write(i+"\r\n")
