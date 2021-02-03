import cv2
import time
import copy
import face_mysql
import tensorflow as tf
import src.facenet
import src.align.detect_face
import numpy as np
from scipy import misc
import calculate
import os
floor = []




get_face = False
count2 = 0



MAX_DISTINCT = 0.9

def image_array_align_data(image_arr, image_path, pnet, rnet, onet, image_size=160, margin=32, gpu_memory_fraction=1.0,
                           detect_multiple_faces=True):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img = image_arr
    # 调用facenet检测人脸
    bounding_boxes, _ = src.align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]

    nrof_successfully_aligned = 0
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        det_arr = []
        img_size = np.asarray(img.shape)[0:2]
        if nrof_faces > 1:
            if detect_multiple_faces:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                img_center = img_size / 2
                offsets = np.vstack(
                    [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                index = np.argmax(
                    bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                det_arr.append(det[index, :])
        else:
            det_arr.append(np.squeeze(det))

        images = np.zeros((len(det_arr), image_size, image_size, 3))
        for i, det in enumerate(det_arr):
            det = np.squeeze(det)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0] - margin / 2, 0)
            bb[1] = np.maximum(det[1] - margin / 2, 0)
            bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
            bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
            cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
            # 进行图片缩放 cv2.resize(origin_image,(w,h))
            scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            nrof_successfully_aligned += 1

            # 保存检测的头像
            filename_base = 'tmp_img_storage'
            filename = os.path.basename(image_path)
            filename_name, file_extension = os.path.splitext(filename)
            # 多个人脸时，在picname后加_0 _1 _2 依次累加。
            output_filename_n = "{}/{}_{}{}".format(filename_base, filename_name, i, file_extension)
            misc.imsave(output_filename_n, scaled)

            scaled = src.facenet.prewhiten(scaled)
            scaled = src.facenet.crop(scaled, False, 160)
            scaled = src.facenet.flip(scaled, False)

            images[i] = scaled
    if nrof_faces > 0:
        return images
    else:
        # 如果没有检测到人脸  直接返回一个1*3的0矩阵  多少维度都行  只要能和是不是一个图片辨别出来就行
        return np.zeros((1, 3))



def FaceDetection(frame, rw=120, rh=90):
    global get_face, count2, outFrame
    global rpi
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    # frame = cv2.flip(frame, 0)
    # frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(gray, 1.3, 6)  # 正脸
    #scaleFactor表示每次图像尺寸减小的比例
    #minNeighbors表示每一个目标至少要被检测到几次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸)

    if len(faces):
        for x, y, h, w in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 255, 0), 2)
    if len(faces) >= 1:
        count2 += 1
    else:
        count2 = 0

    if count2 >= 5:
        count2 = 0
        get_face = True






if __name__=='__main__':
    # with tf.Graph().as_default():
        # gpu_memory_fraction = 1.0
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        # # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        # sess = tf.Session()
        # with sess.as_default():
        #     # 利用MTCNN创建pnet,rnet和onet三个级联
        #     pnet, rnet, onet = src.align.detect_face.create_mtcnn(sess, None)

    pwd = os.getcwd()
    # 当前文件的父路径
    father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
    # 训练模型的路径
    modelpath = '.\\models\\facenet\\20170512-110547'
    with tf.Graph().as_default():
        sess = tf.Session()
        pnet, rnet, onet = src.align.detect_face.create_mtcnn(sess, None)
        # src.facenet.load_model(modelpath)
        # 加载模型
        # 利用facenet加载模型，并分别找出图（metafile）与变量（ckpt_file）
        meta_file, ckpt_file = src.facenet.get_model_filenames(modelpath)
        # 利用tensorflow分别导入图
        saver = tf.train.import_meta_graph(os.path.join(modelpath, meta_file))
        # 利用tensorflow通过session导入变量(ckpt_file)
        saver.restore(sess, os.path.join(modelpath, ckpt_file))
        # 访问模型中的placeholder变量和embedding操作
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    cap = cv2.VideoCapture(1)
    while True:
        ret, frame = cap.read()
        origin_frame = copy.deepcopy(frame)
        FaceDetection(frame)
        cv2.imshow('show',frame)

        cv2.waitKey(1)

        if get_face == True:
            get_face = False
            image_path = r"tmp_img_storage\current.jpg"
            cv2.imwrite(image_path,origin_frame)
            # with open(image_path, 'wb') as f:
            #     f.write(origin_frame)
            #     f.flush()

            img = misc.imread(os.path.expanduser(image_path), mode='RGB')

            images = image_array_align_data(img, image_path, pnet, rnet, onet)

            # 判断如果没有检测到人脸则直接返回
            # if len(images.shape) < 4: return redirect(url_for("face_detect_failed"))
            # 给模型中的placeholder赋新值
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}
            # 在会话中加载操作重新训练
            try:
                emb_array = sess.run(embeddings, feed_dict=feed_dict)
            except:
                continue
            face_query = calculate.matrix()
            # 分别获取距离该图片中人脸最相近的人脸信息
            # pic_min_scores 是数据库中人脸距离（facenet计算人脸相似度根据人脸距离进行的）
            # pic_min_names 是当时入库时保存的文件名
            # pic_min_uid  是对应的用户id
            pic_min_scores, pic_min_names, pic_min_uid, pic_min_floor = face_query.get_socres(emb_array, 'root')

            # 如果提交的query没有group 则返回
            if len(pic_min_scores) == 0: print('group_detect_failed')

            # 设置返回结果
            result = []
            for i in range(0, len(pic_min_scores)):
                if pic_min_scores[i] < MAX_DISTINCT:
                    rdict = {'uid': pic_min_uid[i],
                             'distance': pic_min_scores[i],
                             'pic_name': pic_min_names[i],
                             'floor':pic_min_floor[i]
                             }
                    result.append(rdict)
            print(result)
            if os.path.exists('floor.txt'):
                with open('floor.txt','r') as f:
                    string_tmp = f.read()
                    if len(string_tmp)>0:
                        floor = eval(string_tmp)

            for i in result:
                if isinstance(i['floor'],str) and (eval(i['floor']) not in floor):
                    floor.append(eval(i['floor']))
            print(floor)
            with open('floor.txt','w') as f:
                f.write(str(floor))

            if len(result) == 0:
                print('login_fail')
            else:
                print('welcome')
                # redirect函数一般与url_for函数同步使用；
                # url_for函数第一个参数为函数名称，一般需用装饰器app.route来定义路由






