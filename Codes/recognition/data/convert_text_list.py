with open('char_std_5990.txt', "r", encoding="utf-8") as fd:
    cvt_lines = fd.readlines()

cvt_dict = {}
for i, line in enumerate(cvt_lines):
    value = i
    key = line.strip()
    cvt_dict[key] = value

if __name__ == "__main__":
    train_f = open("train.txt", "w", encoding="utf-8")
    test_f = open("test.txt", "w", encoding="utf-8")
    cvt_file_path = "all_labels.txt"

    with open(cvt_file_path, "r", encoding="utf-8") as fd:
        lines = fd.readlines()
        train = lines[:26000]
        test =lines[26000:30000]
        train_f.writelines(train)
        test_f.writelines(test)