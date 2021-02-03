# -*- coding:utf-8 -*-
"""
利用flask框架构建一个轻型的网站架构，利用tensorflow和facenet来实现人脸识别和比较
用户在用图像登录时会自动检测出人脸，并提取人的128维矩阵向量，同时根据用户的组信息，
与数据库中同组中存储的所有128维向量进行比较，求出欧氏距离，并将其与uid,picname
一同输出在网页上,当欧氏距离小于1的时候认为是一个人，大于1认为不是一个人
"""

from datetime import timedelta

from camera import Camera

"""
利用MTCNN进行人脸识别，并裁剪（只保存识别框中的人脸），利用facenet进行人脸比对，得出欧氏距离，进而判断是否为同一个人
"""
from flask import Flask, request, render_template, url_for, Response
from flask_httpauth import HTTPBasicAuth
# import json

import os

import face_mysql
import tensorflow as tf
import src.facenet
import src.align.detect_face
import numpy as np
from scipy import misc
import calculate

app = Flask(__name__)
# 图片最大为16M
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
auth = HTTPBasicAuth()

# 设置最大的相似距离
MAX_DISTINCT = 0.9

from werkzeug.utils import secure_filename, redirect

# 设置post请求中获取的图片保存的路径
UPLOAD_FOLDER = 'tmp_img_storage/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
else:
    pass
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg','JPG'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


with tf.Graph().as_default():
    gpu_memory_fraction = 1.0
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        # 利用MTCNN创建pnet,rnet和onet三个级联
        pnet, rnet, onet = src.align.detect_face.create_mtcnn(sess, None)

# 当前文件的路径
pwd = os.getcwd()
# 当前文件的父路径
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
# 训练模型的路径
modelpath = '.\\models\\facenet\\20170512-110547'
with tf.Graph().as_default():
    sess = tf.Session()
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

    # 进行人脸识别，加载
    # print('Creating networks and loading parameters')


    # 获取post中的图片并执行插入到库 返回数据库中保存的id
    @app.route('/')
    def index():
        return render_template("face_login.html")

    @app.route('/face_login.html')
    def face_query_html1():
        return render_template("face_login.html")

    @app.route('/face_insert.html')
    def face_insert_html():
        return render_template("face_insert.html")


    @app.route('/face_query.html')
    def face_query_html():
        return render_template("face_login.html")


    @app.route('/face_detect_failed')
    def face_detect_failed():
        return render_template("face_detect_failed.html")


    @app.route('/group_detect_failed')
    def group_detect_failed():
        return render_template('group_detect_failed.html')


    @app.route('/login_failed')
    def login_fail():
        return render_template('login_failed.html')

    @app.route('/face/face_login.html')
    def face_query_html2():
        return render_template("face_login.html")

    @app.route('/face/face_insert.html')
    def face_query_html3():
        return render_template("face_insert.html")

    @app.route('/welcome/<distance>')
    def welcome(distance):
        return render_template('/welcome.html', dist=distance)


    # 获取post中的图片并执行插入到库 返回数据库中保存的id
    @app.route('/face/insert', methods=['POST'])
    def face_insert():
        # 分别获取post请求中的uid 和ugroup作为图片信息
        uid = request.form['uid']
        ugroup = request.form['ugroup']
        upload_files = request.files['imagefile']

        # 从post请求图片保存到本地路径中
        file = upload_files
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(image_path)

        # opencv读取图片，开始进行人脸识别
        img = misc.imread(os.path.expanduser(image_path), mode='RGB')
        # 设置默认插入时 detect_multiple_faces =Flase只检测图中的一张人脸，True则检测人脸中的多张
        # 一般入库时只检测一张人脸，查询时检测多张人脸
        images = image_array_align_data(img, image_path, pnet, rnet, onet, detect_multiple_faces=False)

        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        # emb_array保存的是经过facenet转换的128维的向量
        # emb_array = sess.run(embeddings, feed_dict=feed_dict)
        try:
            # emb_array保存的是经过facenet转换的128维的向量
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
        except:
            return render_template('face_insert.html', request_one='error')
        filename_base, file_extension = os.path.splitext(image_path)
        id_list = []
        # 存入数据库
        for j in range(0, len(emb_array)):
            face_mysql_instant = face_mysql.face_mysql()
            last_id = face_mysql_instant.insert_facejson(filename_base + "_" + str(j),
                                                         ",".join(str(li) for li in emb_array[j].tolist()), uid, ugroup)
            id_list.append(str(last_id))

        # 设置返回类型
        request_result = {}
        request_result['id'] = ",".join(id_list)
        if len(id_list) > 0:
            request_result['state'] = 'sucess'
            return render_template('face_insert.html', request_one=request_result['state'])
        # else:
        #     request_result['state'] = 'error'

        # print(request_result)
        # return json.dumps(request_result)
        # return render_template('face_login.html')

    @app.route('/face/query', methods=['POST'])
    def face_query():

        # 获取查询条件  在ugroup中查找相似的人脸
        ugroup = request.form['ugroup']
        cam = Camera()
        m = cam.get_frame()
        image_path = r"tmp_img_storage\current.jpg"
        with open(image_path, 'wb') as f:
            f.write(m)
            f.flush()
        # upload_files = request.files['imagefile']

        # 获取post请求的图片到本地
        # file = upload_files
        # if file and allowed_file(file.filename):
        #     filename = secure_filename(file.filename)
        #     file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # print(image_path)

        # 读取本地的图片
        img = misc.imread(os.path.expanduser(image_path), mode='RGB')
        images = image_array_align_data(img, image_path, pnet, rnet, onet)

        # 判断如果没有检测到人脸则直接返回
        if len(images.shape) < 4: return redirect(url_for("face_detect_failed"))
        # 给模型中的placeholder赋新值
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        # 在会话中加载操作重新训练
        emb_array = sess.run(embeddings, feed_dict=feed_dict)
        face_query = calculate.matrix()
        # 分别获取距离该图片中人脸最相近的人脸信息
        # pic_min_scores 是数据库中人脸距离（facenet计算人脸相似度根据人脸距离进行的）
        # pic_min_names 是当时入库时保存的文件名
        # pic_min_uid  是对应的用户id
        pic_min_scores, pic_min_names, pic_min_uid = face_query.get_socres(emb_array, ugroup)

        # 如果提交的query没有group 则返回
        if len(pic_min_scores) == 0: return redirect(url_for('group_detect_failed'))

        # 设置返回结果
        result = []
        for i in range(0, len(pic_min_scores)):
            if pic_min_scores[i] < MAX_DISTINCT:
                rdict = {'uid': pic_min_uid[i],
                         'distance': pic_min_scores[i],
                         'pic_name': pic_min_names[i]}
                result.append(rdict)
        print(result)
        if len(result) == 0:
            return redirect(url_for('login_fail'))
        else:
            return redirect(url_for('welcome', distance=result[0]['distance']))
            # redirect函数一般与url_for函数同步使用；
            # url_for函数第一个参数为函数名称，一般需用装饰器app.route来定义路由


# 检测图片中的人脸  image_arr是opencv读取图片后的3维矩阵  返回图片中人脸的位置信息
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


def gen(camera):
    """视频流生成函数"""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """视频流路由(route).放到 img 标签的 src 属性."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080,debug=True)
