import time
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import draw_finger_contours as dr




def trace_fingertip(img: np.ndarray, ratio: float = 0.05, tip_pixel_num: int = 200) -> (float, float):
    """
    识别光幕上指尖位置坐标
    :param img: np.ndarray，二值化的图片，背景的像素值为 255，手指的像素值为 0
    :param ratio: float，图像中黑色像素点个数占总像素点的比率，当大于这个比率就算图像中有手指，否则算没有
    :param tip_pixel_num: int，取指尖最上端的像素点个数，这些像素点坐标的平均值即代表最终识别的指尖的坐标
    :return: tuple，返回指尖的坐标占与图片宽高的比值（相对坐标）
    """
    mask = (img < 127)  # 黑色像素点的 mask
    black_pixel = img[mask]  # 所以黑色像素点
    black_num = black_pixel.shape[0]  # 黑色像素点的个数
    total_num = np.size(img)  # 总像素点个数

    cnt_pixel = 0
    row_start_idx = 0  # 黑色像素点开始的行索引值
    row_end_idx = 0  # 黑色像素点结束的行索引值（取的黑色像素点个数至少为 tip_pixel_num）

    if black_num > 0 and black_num / total_num >= ratio:
        enough_black_pixel = False
        for i in range(img.shape[0] - 1):
            row = img[i, :]
            cnt = np.size(row[(row < 127)])
            if cnt <= 0:
                continue
            if row_start_idx == 0:
                row_start_idx = i
            cnt_pixel += np.size(row[(row < 127)])
            if cnt_pixel > tip_pixel_num:
                row_end_idx = i
                enough_black_pixel = True
                break
        if not enough_black_pixel:
            row_end_idx = img.shape[0] - 1

        # 返回掩模行号 row_start_idx 至 row_end_idx 所有黑色像素点坐标
        coords = np.argwhere(mask[row_start_idx:row_end_idx, :])
        # coords.shape(110,2),shape[0]代表超过tip_pixel_num所需行中所有黑色像素个数，shape[1]代表坐标维度，即2
        if coords.shape[0] != 0 or coords.shape[1] != 0:
            if len(coords) != 0:
                tip = np.mean(coords, 0, dtype=np.int)  # 计算坐标的平均值
                return tip[1] / img.shape[1], (tip[0] + row_start_idx) / img.shape[0]  # (x, y)

    return -1, -1


def darken_key_area_color(keypad_img: np.ndarray, key_area_list: tuple, rate: float = 0.4) -> np.ndarray:
    """
    使某个按键区域变暗，以模拟按键被按下的效果
    :param keypad_img:
    :param key_area:
    :param rate:
    :return:
    """
    for key_area in key_area_list:
        key_area_array = keypad_img[key_area[1]:key_area[3], key_area[0]:key_area[2]]
        zero_array = np.zeros_like(key_area_array)
        dim = cv2.addWeighted(key_area_array, (1 - rate), zero_array, rate, 0)
        keypad_img[key_area[1]:key_area[3], key_area[0]:key_area[2]] = dim
    return keypad_img


def draw_chinese_words(img_array, contents, coord, color=(255, 255, 255), size=40):
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_array)

    # PIL图片上打印汉字
    draw = ImageDraw.Draw(img)  # 图片上打印
    font = ImageFont.truetype("simhei.ttf", size, encoding="utf-8")
    draw.text(coord, contents, color, font=font)

    # PIL 图片转 cv2 图片
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img_array


def main():
    keypad_value = ((13, 14, 15, 16),
                    (9, 10, 11, 12),
                    (5, 6, 7, 8),
                    (1, 2, 3, 4),
                    (-1, -2, -3, -4))
    keypad_area = (
        ((0, 250, 250, 500), (250, 250, 500, 500), (500, 250, 750, 500), (750, 250, 1000, 500)),
        ((0, 500, 250, 750), (250, 500, 500, 750), (500, 500, 750, 750), (750, 500, 1000, 750)),
        ((0, 750, 250, 1000), (250, 750, 500, 1000), (500, 750, 750, 1000), (750, 750, 1000, 1000)),
        ((0, 1000, 250, 1250), (250, 1000, 500, 1250), (500, 1000, 750, 1250), (750, 1000, 1000, 1250)),
        ((0, 1250, 250, 1500), (250, 1250, 500, 1500), (500, 1250, 750, 1500), (750, 1250, 1000, 1500)),
    )  # (left, top, right, bottom)

    pointer_color_flag = 0
    key_row_index, key_column_index = -1, -1
    target_floor = None

    keypad_img_path = 'image/66.png'
    keypad_img = cv2.imread(keypad_img_path)  # keypad_img.shape:(910,1250,3)高度、宽度、通道数
    if keypad_img is None:
        print("keypad image lost!")

    screen_brightness = 0  # 光幕的平均亮度
    # screen_area = ((150, 70), (490, 410))  # (左上角坐标，右下角坐标)
    screen_area = ((40, 0), (600, 480))
    calibrated = False  # 是否校准光幕区域的标志

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('未识别到摄像头')
        return
    prev_key = -5
    start_time = 0.0
    key_record = []
    prev_input_key = -5
    darken_flag = False
    keypad_area_list = []
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        if calibrated:
            # 光幕区域校准后执行以下程序
            # 根据计算的光幕亮度进行图像二值化
            ret, binary = cv2.threshold(gray, screen_brightness - 30, 255,
                                        cv2.THRESH_BINARY)  # binary.shape(480,640)3:4
            binary = binary[screen_area[0][1]:screen_area[1][1],
                     screen_area[0][0]:screen_area[1][0]]  # binary.shape(340,340)
            # print(np.unique(binary))
            tip_coord = trace_fingertip(binary)  # 识别指尖位置

            show_img = np.copy(keypad_img)

            if 0 < tip_coord[0] < 1:
                coord = (int(show_img.shape[1] * tip_coord[0]), int((show_img.shape[0]+250) * tip_coord[1]) + 250)
                # tip_coord是一个相对于正方形的比例，此处img的长宽不相等，因此要进行一定的变化后再乘以比例,140是图像显示部分的高度



                # print('fingertip: ', coord)
                cv2.circle(show_img, coord, 12, (127, 127, 255), 30)  # 画出指尖位置圆点
                x, y = coord
                row, column = -1, -1
                # 键盘按键各区域坐标
                if 250 <= y < 500:
                    row = 0
                elif 500 <= y < 750:
                    row = 1
                elif 750 <= y < 1000:
                    row = 2
                elif 1000 <= y <= 1250:
                    row = 3
                elif 1250 <= y <= 1500:
                    row = 4

                if 0 <= x < 250:
                    column = 0
                elif 250 < x <= 500:
                    column = 1
                elif 500 < x <= 750:
                    column = 2
                elif 750 < x <= 1000:
                    column = 3

                key = keypad_value[row][column]
                # print(key)
                if key == prev_key:
                    curr_time = time.time()
                    if curr_time - start_time > 1:
                        if 0 < key <= 16 or key == -1:
                            if prev_input_key != key:
                                if key not in key_record:
                                    key_record.append(key)
                                prev_input_key = key
                                print(key)
                                pointer_color_flag = 4

                        elif (key == -2) or (key== -3) or (key == -4):
                            if prev_input_key != key:
                                prev_input_key = key
                                pointer_color_flag = 4

                        key_row_index, key_column_index = row, column
                else:
                    prev_input_key = -5
                    prev_key = key
                    start_time = time.time()

            else:
                prev_input_key = -5
                prev_key = -5
                start_time = 0.0

            #新增加的内容：读取摄像头获取的楼层信息
            floor = []
            if os.path.exists('floor.txt'):
                with open('floor.txt', 'r') as f:
                    string_tmp = f.read()
                    if len(string_tmp)>0:
                        floor = eval(string_tmp)
                    print(floor,type(floor))
            if len(floor) > 0:
                for i in floor:
                    if i not in key_record:
                        key_record.append(i)

                        # 此处为获取i在keypad_value元组的索引值，以便来取得i在keypad_area中的索引
                        r = 0
                        index_tmp = ()
                        for k in keypad_value:
                            c = 0
                            for j in k:
                                if j == i:
                                    index_tmp = r, c
                                c += 1
                            r += 1
                        if keypad_area[index_tmp[0]][index_tmp[1]] not in keypad_area_list:
                            keypad_area_list.append(keypad_area[index_tmp[0]][index_tmp[1]])


            if pointer_color_flag > 0:
                if keypad_area[key_row_index][key_column_index] not in keypad_area_list:
                    keypad_area_list.append(keypad_area[key_row_index][key_column_index])
                else:
                    keypad_area_list.remove(keypad_area[key_row_index][key_column_index])
                    if key in key_record:
                        key_record.remove(key)
                    if key in floor:
                        floor.remove(key)
                        with open('floor.txt','w') as f:
                            f.write(str(floor))
                pointer_color_flag = -1






            unique_key_record = list(set(key_record))
            unique_key_record.sort()
            key_record_num = ' '.join([str(x) for x in unique_key_record])  # 转化为集合可去除重复项
            show_img = darken_key_area_color(show_img, keypad_area_list)


            # 在第二个屏中键入数字
            if len(key_record_num) >= 18:
                cut_index = int(len(key_record_num) / 2)  # 当切分点为空格才可切分，以免将数字分开
                if key_record_num[cut_index] == '' and key_record_num[cut_index + 2] != ' ':
                    cut_index = cut_index + 1
                cv2.putText(show_img, key_record_num[:cut_index], (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 128, 0), 2)
                cv2.putText(show_img, key_record_num[cut_index:], (300, 190), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 128, 0), 2)
            else:
                cv2.putText(show_img, key_record_num, (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 128, 0), 2)

            # 在第一个屏中键入数字
            if 0 < tip_coord[0] < 1:
                cv2.putText(show_img, str(key), (95, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 128, 0), 2)

            if target_floor is not None:
                cv2.putText(show_img, target_floor, (875, 110), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 128, 0), 6)
            cv2.namedWindow('fingertip', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('fingertip',400,600)

            cv2.imshow('fingertip', show_img)
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

            contours = dr.draw_finger_contours(frame)
            cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)
            frame = cv2.rectangle(frame, screen_area[0], screen_area[1], (200, 200, 0), 2)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            print(floor)

        else:
            # 在完成光幕区域校准之前执行以下程序
            # 校准光幕放置区域
            img = cv2.rectangle(frame, screen_area[0], screen_area[1], (200, 200, 0), 2)
            cv2.namedWindow('calibrate', cv2.WINDOW_NORMAL)
            cv2.imshow('calibrate', img)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                # 取中心 40*40 的像素样本计算光幕平均亮度
                sample = gray[220:260, 300:340]
                screen_brightness = int(np.mean(sample))  # 给光幕亮度变量赋值
                print('screen brightness: ', screen_brightness)
                calibrated = True  # 设置校准标志为已校准完成
                cv2.destroyWindow('calibrate')
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()
