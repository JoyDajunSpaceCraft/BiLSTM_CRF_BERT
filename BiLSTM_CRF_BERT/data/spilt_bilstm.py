bilist = []
with open("/Users/apple/PycharmProjects/BiLSTM_CRF_BERT/BiLSTM_CRF_BERT/data/BILSTM_BERT_CRF.txt", "r") as f:
    chunk = []
    start_flag = False
    for item in f.readlines():
        if item != "\n" and item.split(" ")[2] == "B\n":
            chunk.append(item.split(" ")[0])
            start_flag = True
            print()
        elif item != "\n" and item.split(" ")[2] == "I\n":
            chunk.append(item.split(" ")[0])
        elif item != "\n" and item.split(" ")[2] == "O\n" and start_flag == True:
            bilist.append(chunk)
            chunk = []
            start_flag = False
        else:
            continue
print(bilist)

with open("BiList.txt", "a") as add_file:
    for item in bilist:
        seq = ""
        if len(item) > 1:
            for index, value in enumerate(item):
                if index != len(item) - 1:
                    seq += value + " "
                else:
                    seq += value
        else:
            seq = item[0]
        add_file.write(seq + "\r\n")
