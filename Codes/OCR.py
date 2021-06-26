# -*- coding:utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from torch.autograd import Variable
import detection.config as Dconfig
import recognition.lib.config.alphabets as alphabets
import recognition.lib.models.crnn as crnn
import recognition.lib.utils.utils as utils
from detection.ctpn_model import CTPN_Model
from detection.ctpn_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox, nms, \
    TextProposalConnectorOriented
from PIL import ImageFont, ImageDraw, Image
from cut_img import get_rotate_img

img_path = 'detection/images/demo/chinese1.jpg'                 #测试图片
model_ctpn_path = "detection/checkpoints/vgg16.pth"             #ctpn模型
model_crnn_path = "recognition/output/checkpoints/crnn.pth"   #crnn模型



def parse_arg():
    cfg = "recognition/lib/config/config.yaml"
    with open(cfg, 'r') as f:
        Rconfig = yaml.load(f)
        Rconfig = edict(Rconfig)

    Rconfig.DATASET.ALPHABETS = alphabets.alphabet
    Rconfig.MODEL.NUM_CLASSES = len(Rconfig.DATASET.ALPHABETS)

    return Rconfig


def dis(image, s):
    h, w = image.shape[0:2]
    img = cv2.resize(image, (int(w / 1.3), int(h / 1.3)))
    cv2.imshow(f'image_{s}', img)
    new_path = img_path[:-4] + f"_demo_{s}.jpg"
    print(new_path)
    cv2.imwrite(new_path, image)


def recognition(config, img, model, converter, device):
    h, w = img.shape

    img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=config.MODEL.IMAGE_SIZE.H / h,
                     interpolation=cv2.INTER_CUBIC)

    h, w = img.shape
    w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    pred = model(img)

    _, pred = pred.max(2)
    pred = pred.transpose(1, 0).contiguous().view(-1)

    pred_size = Variable(torch.IntTensor([pred.size(0)]))
    sim_pred = converter.decode(pred.data, pred_size.data, raw=False)

    return sim_pred


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    prob_thresh = 0.7
    width = 1000
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weights = os.path.join(model_ctpn_path)

    model_ctpn = CTPN_Model()
    model_ctpn.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
    model_ctpn.to(device)
    model_ctpn.eval()

    Rconfig = parse_arg()

    model_crnn = crnn.get_crnn(Rconfig).to(device)
    print('loading pretrained model from {0}'.format(model_crnn_path))
    checkpoint = torch.load(model_crnn_path)
    if 'state_dict' in checkpoint.keys():
        model_crnn.load_state_dict(checkpoint['state_dict'])
    else:
        model_crnn.load_state_dict(checkpoint)

    converter = utils.strLabelConverter(Rconfig.DATASET.ALPHABETS)

    image = cv2.imread(img_path)
    x, y = image.shape[0:2]
    if y > 1500:
        scale = y / 1500
    else:
        scale = 1

    image = cv2.resize(image, (int(y / scale), int(x / scale)))
    image_c = image.copy()
    h, w = image.shape[:2]
    image = image.astype(np.float32) - Dconfig.IMAGE_MEAN
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()

    with torch.no_grad():
        image = image.to(device)
        cls, regr = model_ctpn(image)
        cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
        regr = regr.cpu().numpy()
        anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
        bbox = bbox_transfor_inv(anchor, regr)
        bbox = clip_box(bbox, [h, w])

        fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
        select_anchor = bbox[fg, :]
        select_score = cls_prob[0, fg, 1]
        select_anchor = select_anchor.astype(np.int32)

        keep_index = filter_bbox(select_anchor, 16)

        # nsm
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nmsbox = np.hstack((select_anchor, select_score))
        keep = nms(nmsbox, 0.3)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        # text line-
        textConn = TextProposalConnectorOriented()
        text = textConn.get_text_lines(select_anchor, select_score, [h, w])
        # print(text)
        img_copy = image_c.copy()
        img_text = np.zeros((h, w, 3), np.uint8)
        # 使用白色填充图片区域,默认为黑色
        img_text.fill(255)
        result = []
        for i in text:
            s = str(round(i[-1] * 100, 2)) + '%'
            i = [int(j) for j in i]
            x_min = min(i[0], i[4]) - 3
            x_max = max(i[2], i[6])
            y_max = max(i[5], i[7])
            y_min = min(i[1], i[3])
            region = get_rotate_img(img_copy,i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7])
            # region = img_copy[y_min:y_max, x_min:x_max]
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            res = recognition(Rconfig, region, model_crnn, converter, device)

            cv2.line(image_c, (x_min, y_min), (x_max, y_min), (180, 0, 0), 1)
            cv2.line(image_c, (x_min, y_min), (x_min, y_max), (0, 0, 180), 1)
            cv2.line(image_c, (x_max, y_max), (x_max, y_min), (0, 0, 180), 1)
            cv2.line(image_c, (x_min, y_max), (x_max, y_max), (180, 0, 0), 1)
            fontpath = "simsunttc/simsun.ttc"  # <== 这里是宋体路径
            font = ImageFont.truetype(fontpath, 20)
            img_pil = Image.fromarray(img_text)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x_min, y_min + 5), res, font=font, fill=(255, 0, 0, 0))  # s+":"+
            img_text = np.array(img_pil)

            cv2.putText(image_c, s, (i[0] + 13, i[1] + 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (180, 0, 0),
                        1,
                        cv2.LINE_AA)
            result.append({"text": res, "x": x_min, "y": y_min})

        result.sort(key=lambda x: (x["y"], x["x"]))
        for i in result:
            print(i["text"])
        dis(image_c, "a")
        dis(img_text, "b")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
