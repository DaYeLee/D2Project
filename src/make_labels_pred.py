
# -*- coding: utf-8 -*-

"""Inception v3 architecture 모델을 retraining한 모델을 이용해서 이미지에 대한 추론(inference)을 진행하는 예제"""

import numpy as np
import tensorflow as tf
import os
from config import *
from evaluation import *

#modelFullPath = '/tmp/output_graph.pb'  # 읽어들일 graph 파일 경로
#labelsFullPath = '/tmp/output_labels.txt'  # 읽어들일 labels 파일 경로

modelFullPath = OUTPUT_GRAPH           # 읽어들일 graph 파일 경로
labelsFullPath = OUTPUT_LABELS          # 읽어들일 labels 파일 경로


def create_graph():
    """저장된(saved) GraphDef 파일로부터 graph를 생성하고 saver를 반환한다."""
    # 저장된(saved) graph_def.pb로부터 graph를 생성한다.
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():

    # get list of files
    if os.path.exists(IMG_DIR) is False:
        print("Folder %s not found, copy image files to %s" % (IMG_DIR, IMG_DIR))
        return

    img_path = os.listdir(IMG_DIR)
    img_path.sort()

    tmpcnt = -1
    for i in range(len(img_path)):
        # 이미지 파일의 절대 경로
        img_paths = IMG_DIR + "/" + img_path[i]

        # img/dummy/ 내부의 숨김파일 예외처리
        tmp = img_paths
        tmp2 = tmp.partition('/dummy/')
        tmp3 = tmp2[2]
        if tmp3[0] == '.':
            continue

        if not tf.gfile.Exists(img_paths):
            tf.logging.fatal('File does not exist %s', img_paths)
            return None

        # 이미지 파일 읽어오기
        image_data = tf.gfile.FastGFile(img_paths, 'rb').read()

        # 저장된(saved) GraphDef 파일로부터 graph를 생성한다.
        create_graph()

        with tf.Session() as sess:
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions = sess.run(softmax_tensor,
                                   {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)
            top_k = predictions.argsort()[-5:][::-1]  # 가장 높은 확률을 가진 5개(top 5)의 예측값(predictions)을 얻는다.
            f = open(labelsFullPath, 'rb')
            lines = f.readlines()
            labels = [str(w).replace("\n", "") for w in lines]


        if os.path.exists(DATA_DIR) is False:
            os.makedirs(DATA_DIR)
        if tmpcnt == -1:
            with open(os.path.join(DATA_DIR, LABELS_PRED + ".txt"), 'w') as ff:
                ff.writelines(labels[top_k[0]]+"\n")
                #ff.writelines([line + "\n" for line in labels[top_k[0]]])
        else:
            with open(os.path.join(DATA_DIR, LABELS_PRED + ".txt"), 'a') as ff:
                ff.writelines(labels[top_k[0]] + "\n")
        tmpcnt = 1

    answerpath = DATA_DIR + "/labels_true.txt"
    predictpath = DATA_DIR + "/labels_pred.txt"
    evaluation(answerpath, predictpath)

if __name__ == '__main__':
    run_inference_on_image()