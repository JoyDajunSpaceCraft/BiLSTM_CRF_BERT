# with open("text.txt", "w") as new:
#     with open("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/test.txt", "r") as f:
#         for i in f.readlines():
#             if i == "\t" + "O":
#                 continue
#             else:
#                 new.write(i)

def count(filename):
    CR = 0
    CPR = 0
    IPR = 0

    with open(filename, "r") as f:
        for item in f.readlines():
            if item != "\n" and len(item.split("    ")) > 1:
                define = item.split("    ")[1]
                if define == "CR\n":
                    CR += 1
                elif define == "IPR\n":
                    IPR += 1
                elif define == "CPR\n":
                    CPR += 1
                else:
                    continue

    print("CR:{},CPR:{},IPR:{}".format(CR, CPR, IPR))


if __name__ == "__main__":
    count("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/sci_bert_new.txt")