# 孙浩博
# 2021/4/7 10:03
import cv2
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.decomposition import PCA
import time


def get_train_and_test_img_features():
    train_path = './training/'
    test_path = './testing/'
    train_feature = []
    test_feature = []

    train_img_list = os.listdir(train_path)
    test_img_list = os.listdir(test_path)

    for train_img in train_img_list:
        img = cv2.imread(train_path + train_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度化
        equalize = cv2.equalizeHist(gray)  # 预处理，灰度均值化
        kp, des = get_sift_features(equalize)
        train_feature.append(des)
    for test_img in test_img_list:
        img = cv2.imread(test_path + test_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalize = cv2.equalizeHist(gray)
        kp, des = get_sift_features(equalize)
        test_feature.append(des)
    return train_feature, test_feature


def PCA_feature(train_feature, test_feature, component):
    transfer = PCA(n_components=component)
    new_train_feature = []
    new_test_feature = []
    for i, feature in enumerate(train_feature):
        transfer.fit(train_feature[i])
        new_train_feature.append(transfer.transform(train_feature[i]))
        new_test_feature.append(transfer.transform(test_feature[i]))

    return new_train_feature, new_test_feature


def get_sift_features(img, type='sift'):
    if type == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
    elif type == 'surf':
        sift = cv2.xfeatures2d.SURF_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


def sift_detect_match_num(des, des_train, ratio=0.75):
    # 使用KNN计算查询图像与训练图像之间匹配的点数目,采用k=2近邻匹配，最近的点距离与次近点距离之比小于阈值ratio就认为是成功匹配。
    # bf = cv2.BFMatcher(cv2.NORM_L1)
    bf = cv2.BFMatcher()
    # 一次匹配的数量sift_nb_1∗sift_nb_2
    matches = bf.knnMatch(des, des_train, k=2)
    match_num = 0
    for first, second in matches:
        if first.distance < ratio * second.distance:
            match_num = match_num + 1
    return match_num


def get_one_palm_match_num(des, index, train_feature, ratio=0.75):
    # 获取查询图像与训练图像中属于每一组的3张图像的匹配点的数量和
    match_num_sum = 0
    for i in range(index, index + 3):
        match_num_sum += sift_detect_match_num(des, train_feature[i], ratio=ratio)
    return match_num_sum


def get_match_result(des, train_feature, ratio=0.75):
    # 根据最大的匹配点数量和，确定查询图片的类别
    index = 0
    train_length = len(train_feature)
    result = np.zeros(train_length // 3, dtype=np.int32)
    while index < train_length:
        result[index // 3] = get_one_palm_match_num(des, index, train_feature, ratio=ratio)
        index += 3
    return result.argmax()


def predict(train_features, test_features, ratio=0.75, component=0):
    # 预测正确率
    predict_true = 0
    for i, feature in enumerate(test_features):
        print('Processing image', i + 1, '...')
        # 预测每张测试图片的类别
        category = get_match_result(feature, train_features, ratio=ratio)
        if category == i // 3:
            predict_true += 1
        print('Predict result:', category + 1, 'Groud truth:', i // 3 + 1)
    if component == 0:
        print('Predict the correct number of pictures:', predict_true, 'Accuracy:', predict_true / len(test_features),
              'ratio:', ratio)
    elif component > 0:
        print('Predict the correct number of pictures:', predict_true, 'Accuracy:', predict_true / len(test_features),
              'component:', component)
    return predict_true / len(test_features)


def show_plot(x, y, name, title):
    # 绘制准确率变化图
    plt.figure()
    plt.plot(x, y, color='red')
    plt.title(title)
    plt.savefig('./Image_result/{0}.png'.format(name))
    plt.show()


def find_best_ratio(ratio=0.65, max_ratio=0.8):
    # 存储每张图片的SIFT特征描述向量
    train_sift_features, test_sift_features = get_train_and_test_img_features()

    # 初始比率
    ratio = 0.65
    # 正确率
    best_acc = 0
    best_ratio = 0
    ratio_list = []
    acc_list = []
    # 最大比率
    max_ratio = 0.8
    while ratio <= max_ratio:  # 循环测试具有最高准确率的ratio
        acc = predict(train_sift_features, test_sift_features, ratio)
        acc_list.append(acc)
        ratio_list.append(ratio)
        if acc > best_acc:
            best_acc = acc
            best_ratio = ratio
        ratio += 0.01
    title = 'best ratio:' + str(best_ratio) + " best acc:{:.4f}".format(best_acc)
    plt_name = "ratio=[{0},{1}]".format(ratio, max_ratio)
    show_plot(ratio_list, acc_list, plt_name, title)
    print(title)


def find_best_component():
    # 存储每张图片的SIFT特征描述向量
    train_sift_features, test_sift_features = get_train_and_test_img_features()
    # 初始比率
    ratio = 0.7
    # 初始降维
    component = 40
    # 正确率
    best_acc = 0
    best_component = 0
    component_list = []
    acc_list = []
    # 最大维数
    max_component = 55
    while component <= max_component:  # 循环测试具有最高准确率的component
        # 加入PCA
        train_features, test_features = PCA_feature(train_sift_features, test_sift_features, component)

        acc = predict(train_features, test_features, ratio, component)
        acc_list.append(acc)
        component_list.append(component)
        if acc > best_acc:
            best_acc = acc
            best_component = component
        component += 1
    title = 'best_component:' + str(best_component) + " best acc:{:.4f}".format(best_acc)
    plt_name = "ration=0.7 best_component=[{0},{1}]".format(component, max_component)
    show_plot(component_list, acc_list, plt_name, title)
    print(title)


def main():
    # ratio = input("Enter the initial ratio\n")
    # max_ratio = input("Enter the max ratio\n")
    # item = input("Whether to choose PCA dimensionality reduction?\nyes or no\n")
    # if item == 'yes':
    #     find_best_component()
    # else:
    #     find_best_ratio(ratio, max_ratio)
    find_best_ratio(0.65, 0.8)


if __name__ == '__main__':
    Start_time = time.time()
    main()
    End_time = time.time()
    print("Total time:" + str(End_time - Start_time))
