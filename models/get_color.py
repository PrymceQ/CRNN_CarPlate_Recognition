import cv2
import numpy as np

PLATE_COLOR_RGB_DICT = {
    '蓝': [0, 0, 255],
    '绿': [0, 255, 0],
    '黄': [255, 255, 0],
    '黑': [0, 0, 0],
    '白': [255, 255, 255],
}

def get_main_color(image):
    # BGR -> RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 对所有的像素（3通道）进行K-Means聚类
    pixels = image_rgb.reshape((-1, 3))

    k = 3 # 聚类中心数量
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 计算每个类别的像素数量
    unique, counts = np.unique(labels, return_counts=True)
    counts = counts / sum(counts)

    # 找到并输出主要的颜色rgb值
    main_color_label = unique[np.argmax(counts)]
    main_color_rgb = centers[main_color_label].astype(int)

    return main_color_rgb

def calculate_euclidean_distance(color1, color2):
    return np.linalg.norm(np.array(color1) - np.array(color2))

def find_nearest_color(rgb_value, color_dict):
    min_distance = float('inf')
    nearest_color = None
    for color, color_rgb in color_dict.items():
        distance = calculate_euclidean_distance(rgb_value, color_rgb)
        if distance < min_distance:
            min_distance = distance
            nearest_color = color
    return nearest_color

def color(bgr_image):
    # 获取主要颜色rgb值
    main_color = get_main_color(bgr_image)
    # print(f"主要颜色的RGB值： {main_color}")

    # 根据距离得到主要颜色的类别
    color_class = find_nearest_color(main_color, PLATE_COLOR_RGB_DICT)
    # print(f"主要颜色为： {color_class}")

    return main_color, color_class


if __name__=="__main__":

    image_path = '../test_images/1.jpg'
    image = cv2.imread(image_path)
    main_color, color_class = color(bgr_image=image)

