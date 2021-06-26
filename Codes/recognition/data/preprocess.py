import glob, os
import cv2 as cv
from cut_img import cut_img
from fractions import Fraction


# paths.sort()

def txt2std():
    cvt_dict = {}
    txt_f1 = open('train2.txt', "r", encoding="utf-8")
    txt_f2 = open('test2.txt', "r", encoding="utf-8")
    std_f = open('char_std_5990.txt', "r", encoding="utf-8")
    data1=txt_f1.readlines()
    data2=txt_f2.readlines()
    data3=std_f.readlines()
    print(len(data1),len(data2))
    for line in data1:
        cvt_lines = line.split()[-1]
        for i, word in enumerate(cvt_lines):
            key = word
            value = i
            cvt_dict[key] = value
    for line in data2:
        cvt_lines = line.split()[-1]
        for i, word in enumerate(cvt_lines):
            key = word
            value = i
            cvt_dict[key] = value

    for i, word in enumerate(data3):
        word = word.split()[0]
        key = word
        value = i
        cvt_dict[key] = value

    print(len(cvt_dict.keys()))
    py_fd = open("../lib/config/alphabets_6002.py", "w", encoding="utf-8")
    alphabet = "".join(cvt_dict.keys())
    py_fd.write("alphabet = \"\"\"" + alphabet + "\"\"\"")


txt2std()


def char_std2py():
    py_fd = open("../lib/config/alphabets_std.py", "w", encoding="utf-8")
    with open('char_std_5990.txt', "r", encoding="utf-8") as fd:
        lines = fd.readlines()
        words = ("".join(lines)).strip().replace('\n', '').replace('\r', '')
        print(len(words))
        py_fd.write("alphabet = \"\"\"" + words + "\"\"\"")


#char_std2py()


def chinese_img_process():
    new_images_path = "F:/chinese-text_detection/OCR_CN/recognition/data/cn_image/"
    paths = glob.glob(("F:/chinese-text_detection/datasets/chinese-images/" + '*.jpg'))
    for path in paths[:30000]:
        img = cv.imread(path)
        jpg_name = new_images_path + path.split("\\")[-1]
        print(jpg_name)
        cv.imwrite(jpg_name, img)


def img2small():
    new_images_path = "train_image/"
    images_path = "train_1000/image_1000/"
    labels_path = "train_image_labels.txt"
    paths = glob.glob(("train_1000/txt_1000" + '/*.txt'))
    index = 0
    labels = []
    for path in paths:
        image_path = path.split("\\")[-1]
        image_name = image_path[:-3] + "jpg"
        print(image_name)
        fw = open(labels_path, "w", encoding="utf-8")
        with open(path, "r", encoding="utf-8") as fd:
            logs = fd.readlines()
            for log in logs:
                images = log.split(",")[:-1]
                words = log.split(",")[-1]
                if "#" not in words:
                    index += 1
                    print(words, end="")
                    cut_img(images_path, image_name, new_images_path,
                            round(float(images[0])), round(float(images[1])),
                            round(float(images[2])), round(float(images[3])),
                            round(float(images[4])), round(float(images[5])),
                            round(float(images[6])), round(float(images[7])),
                            index
                            )
                    labels.append(words)

        fw.writelines(labels)


def img_clear():
    images_path = "cn_image/"
    new_labels_path = "labels.txt"
    paths = open("train_image_labels.txt", "r", encoding="utf-8").readlines()
    fw = open(new_labels_path, "w", encoding="utf-8")
    clear_labels = []
    X, Y = 0, 0
    """
    for i, label in enumerate(paths):

        image_name = label.split()[0]

        img = cv.imread(images_path + image_name)
        x, y = img.shape[0],img.shape[1]
        print(x,y)
        if x > X:
            X = x
        if y > Y:
            Y = y
    print(X, Y)
    """

    for i, label in enumerate(paths):
        image_name = "_" + str(i + 1) + ".jpg"
        img = cv.imread(images_path + image_name)
        if img is not None:
            x = img.shape[0]
            y = img.shape[1]
            if Fraction(x, y) < Fraction(1, 4) and Fraction(x, y) > Fraction(1, 7):
                print(image_name, "|", x, y)
                new_label = image_name + " " + label
                clear_labels.append(new_label)

    fw.writelines(clear_labels)


# img_clear()


def new_train_label():
    f_tr = open("train.txt", "r", encoding="utf-8")
    f_te = open("test.txt", "r", encoding="utf-8")
    f_new_tr = open("train_label.txt", "w", encoding="utf-8")
    f_new_te = open("test_label.txt", "w", encoding="utf-8")
    fr = open("labels.txt", "r", encoding="utf-8")
    train = f_tr.readlines()
    test = f_te.readlines()
    news = fr.readlines()
    print(len(train), len(test), len(news))
    train += news[:200]
    test += news[200:362]
    f_new_tr.writelines(train)
    f_new_te.writelines(test)


# new_train_label()
def renew_label():
    f_t = open("all_labels.txt", "r", encoding="utf-8")
    f_new_t1 = open("train2.txt", "w", encoding="utf-8")
    f_new_t2 = open("test2.txt", "w", encoding="utf-8")
    #fr = open("labels.txt", "r", encoding="utf-8")
    data = f_t.readlines()
    train = data[:500000]
    test = data[500000:506000]
    #news = fr.readlines()
    print(len(train), len(test))#, len(news)
    #train += news[:200]
    #test += news[200:362]
    f_new_t1.writelines(train)
    f_new_t2.writelines(test)

#renew_label()
