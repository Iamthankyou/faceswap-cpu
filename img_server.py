
import paddle
import argparse
import cv2
import numpy as np
import os
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel

from multiprocessing import Process


from flask import Flask, jsonify
import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, send_file
from werkzeug.utils import secure_filename
import sys
from PIL import Image

#START VIDEO

import paddle
import argparse
import cv2
import numpy as np
import os
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel
from tqdm import tqdm

def get_id_emb2(id_net, id_img):
    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2paddle(id_img)
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std

    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)

    return id_emb, id_feature


def video_test(args):

    paddle.set_device("gpu" if args.use_gpu else 'cpu')
    faceswap_model = FaceSwap(args.use_gpu)

    id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
    id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))

    id_net.eval()

    weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')

    landmarkModel = LandmarkModel(name='landmarks')
    landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    id_img = cv2.imread(args.source_img_path)

    landmark = landmarkModel.get(id_img)
    if landmark is None:
        print('**** No Face Detect Error ****')
        exit()
    aligned_id_img, _ = align_img(id_img, landmark)

    id_emb, id_feature = get_id_emb2(id_net, aligned_id_img)

    faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
    faceswap_model.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture()
    cap.open(args.target_video_path)
    videoWriter = cv2.VideoWriter(os.path.join(args.output_path, os.path.basename(args.target_video_path)), fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    all_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in tqdm(range(int(all_f))):
        ret, frame = cap.read()
        landmark = landmarkModel.get(frame)
        if landmark is not None:
            att_img, back_matrix = align_img(frame, landmark)
            att_img = cv2paddle(att_img)
            res, mask = faceswap_model(att_img)
            res = paddle2cv(res)
            mask = np.transpose(mask[0].numpy(), (1, 2, 0))
            res = dealign(res, frame, back_matrix, mask)
            frame = res
        else:
            print('**** No Face Detect Error ****')
        videoWriter.write(frame)
    cap.release()
    videoWriter.release()


def video_test2():
    source_img_path = 'results/src.png'
    target_video_path = 'results/dst.mp4'
    output_path = 'ionic-multi-menu/src/assets/images/demo'
    image_size = 224
    merge_result = True
    use_gpu = False

    paddle.set_device("gpu" if use_gpu else 'cpu')
    faceswap_model = FaceSwap(use_gpu)

    id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
    id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))

    id_net.eval()

    weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')

    landmarkModel = LandmarkModel(name='landmarks')
    landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    id_img = cv2.imread(source_img_path)

    landmark = landmarkModel.get(id_img)
    if landmark is None:
        print('**** No Face Detect Error ****')
        exit()
    aligned_id_img, _ = align_img(id_img, landmark)

    id_emb, id_feature = get_id_emb2(id_net, aligned_id_img)

    faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
    faceswap_model.eval()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    cap = cv2.VideoCapture()
    cap.open(target_video_path)
    videoWriter = cv2.VideoWriter(os.path.join(output_path, os.path.basename(target_video_path)), fourcc, int(cap.get(cv2.CAP_PROP_FPS)), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    all_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for i in tqdm(range(int(all_f))):
        ret, frame = cap.read()
        landmark = landmarkModel.get(frame)
        if landmark is not None:
            att_img, back_matrix = align_img(frame, landmark)
            att_img = cv2paddle(att_img)
            res, mask = faceswap_model(att_img)
            res = paddle2cv(res)
            mask = np.transpose(mask[0].numpy(), (1, 2, 0))
            res = dealign(res, frame, back_matrix, mask)
            frame = res
        else:
            print('**** No Face Detect Error ****')
        videoWriter.write(frame)
    cap.release()
    videoWriter.release()

    print("CONVERT VIDEO")
    os.system(" ffmpeg -i ionic-multi-menu/src/assets/images/demo/dst.mp4 -vcodec libx264 -f mp4 ionic-multi-menu/src/assets/images/demo/output.mp4")
    print("END CONVERT VIDEO")

def auto_swap_video():
    video_test2()
#END VIDEO


from flask_cors import CORS, cross_origin

UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))+'/results/'
# UPLOAD_FOLDER = os.path.dirname(os.path.abspath(__file__))+'/ionic-multi-menu/src/assets/images'

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'}

application = Flask(__name__, static_url_path="/static")
CORS(application)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# limit upload size upto 8mb
application.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024


def get_id_emb(id_net, id_img_path):
    id_img = cv2.imread(id_img_path)

    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2paddle(id_img)
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std

    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)

    return id_emb, id_feature

def image_test_auto1(target_img_path, source_img_path, output_dir):
    merge_result = True

    print("Image auto merge")

    # paddle.set_device("gpu" if args.use_gpu else 'cpu')
    # False -> CPU
    faceswap_model = FaceSwap(False)

    id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
    id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))

    id_net.eval()

    weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')

    base_path = source_img_path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
    id_emb, id_feature = get_id_emb(id_net, base_path + '_aligned.png')

    faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
    faceswap_model.eval()

    if os.path.isfile(target_img_path):
        img_list = [target_img_path]
    else:
        img_list = [os.path.join(target_img_path, x) for x in os.listdir(target_img_path) if
                    x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]

    for img_path in img_list:
        origin_att_img = cv2.imread(img_path)
        base_path = img_path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        att_img = cv2.imread(base_path + '_aligned.png')
        att_img = cv2paddle(att_img)
        import time

        res, mask = faceswap_model(att_img)
        res = paddle2cv(res)

        if merge_result:
            back_matrix = np.load(base_path + '_back.npy')
            mask = np.transpose(mask[0].numpy(), (1, 2, 0))
            res = dealign(res, origin_att_img, back_matrix, mask)
        cv2.imwrite(os.path.join(output_dir, os.path.basename(img_path)), res)


def face_align(landmarkModel, image_path, merge_result=False, image_size=224):
    if os.path.isfile(image_path):
        img_list = [image_path]
    else:
        img_list = [os.path.join(image_path, x) for x in os.listdir(image_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
    for path in img_list:
        img = cv2.imread(path)
        landmark = landmarkModel.get(img)
        if landmark is not None:
            base_path = path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
            aligned_img, back_matrix = align_img(img, landmark, image_size)
            # np.save(base_path + '.npy', landmark)
            cv2.imwrite(base_path + '_aligned.png', aligned_img)
            if merge_result:
                np.save(base_path + '_back.npy', back_matrix)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# application = Flask(__name__)

@application.route("/upload_src", methods=['GET', 'POST'])
@cross_origin(origin='localhost:4200')
def upload_src():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        file.filename = 'src.png'

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))

        path = (os.path.join(application.config['UPLOAD_FOLDER'], filename))
        print("path :", path)

        result = path.split("/")
        filename2 = result[-1:]
        print("fname :", filename2)
        filename1 = " ".join(filename2)

    return "OK"

@application.route("/upload_dst", methods=['GET', 'POST'])
@cross_origin(origin='localhost:4200')
def upload_dst():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        file.filename = 'dst.png'

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))

        path = (os.path.join(application.config['UPLOAD_FOLDER'], filename))
        print("path :", path)

        result = path.split("/")
        filename2 = result[-1:]
        print("fname :", filename2)
        filename1 = " ".join(filename2)

        print("Done")
        res_file = './results/dst.png'

        target_img_path = './results/dst.png'
        source_img_path = './results/src.png'

        landmarkModel = LandmarkModel(name='landmarks')
        landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
        face_align(landmarkModel, source_img_path)
        face_align(landmarkModel, target_img_path, True, 224)

        target_img_path = './results/dst.png'
        source_img_path = './results/src.png'
        output_dir = './results/res'

        image_test_auto1(target_img_path, source_img_path, output_dir)

        return send_file(res_file, mimetype='image/gif')

    # return render_template('index1.html')


@application.route("/upload_video", methods=['GET', 'POST'])
@cross_origin(origin='localhost:4200')
def upload_video():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            print('No file selected')
            return redirect(request.url)

        file.filename = 'dst.mp4'

        print("POST")

        if file:
            print("File and Allow file")
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))

            path = (os.path.join(application.config['UPLOAD_FOLDER'], filename))
            print("path :", path)

            result = path.split("/")
            filename2 = result[-1:]
            print("fname :", filename2)
            filename1 = " ".join(filename2)

            print("Done")
            res_file = './results/dst.png'

            print("Start swap video")
            auto_swap_video()
            print("End swap video")

            return "OK"

    # return render_template('index1.html')

@application.route("/swap_style", methods=['GET'])
@cross_origin(origin='localhost:4200')
def swap_style():
    args = request.args
    name = args.get('name')

    name = name.split('=')[1]

    target_img_path = './results/style/' + name
    print('Name of img style ' + name)

    output_dir = 'results'
    source_img_path = './results/src.png'

    landmarkModel = LandmarkModel(name='landmarks')
    landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    face_align(landmarkModel, source_img_path)
    face_align(landmarkModel, target_img_path, True, 224)


    output_dir = './results/styleout'

    image_test_auto1(target_img_path, source_img_path, output_dir)

    res_file = './results/dst.png'
    return send_file(res_file, mimetype='image/gif')


@application.route("/swap_images", methods=['GET'])
@cross_origin(origin='localhost:4200')
def swap_image():
    target_img_path = './results/dst.png'
    output_dir = 'results'
    source_img_path = './results/src.png'

    landmarkModel = LandmarkModel(name='landmarks')
    landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    face_align(landmarkModel, source_img_path)
    face_align(landmarkModel, target_img_path, True, 224)

    target_img_path = './results/dst.png'
    source_img_path = './results/src.png'
    output_dir = './results/res'

    image_test_auto1(target_img_path, source_img_path, output_dir)

    res_file = './results/dst.png'
    return send_file(res_file, mimetype='image/gif')

@application.route('/search', methods=['GET'])
@cross_origin(origin='localhost:4200')
def search():
    # print("GO")
    args = request.args
    name = args.get('name')
    res_file = './results/styleout/' + name
    return send_file(res_file, mimetype='image/gif')

@application.route('/search2', methods=['GET'])
@cross_origin(origin='localhost:4200')
def search2():
    # print("GO")
    args = request.args
    name = args.get('name')
    res_file = './results/style/' + name
    return send_file(res_file, mimetype='image/gif')

@application.route("/get_number", methods=['GET', 'POST'])
@cross_origin(origin='localhost:4200')
def get_number():
    styles = os.listdir("./results/style")
    index = 0

    list = []

    for i in range(0, len(styles)):
        if "jpg" in styles[i]:
            index+=1
            list.append(styles[i])

    return jsonify(result=list)

@application.route("/hair_style2", methods=['GET', 'POST'])
@cross_origin(origin='localhost:4200')
def hair_style2():
    iter = 0
    print("START HAIR STYLE")
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        file.filename = 'src.png'

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))

        path = (os.path.join(application.config['UPLOAD_FOLDER'], filename))
        print("path :", path)

        result = path.split("/")
        filename2 = result[-1:]
        print("fname :", filename2)
        filename1 = " ".join(filename2)

        print("Done Upload")

    styles = os.listdir("./results/style")
    #
    # for i in range (0,len(styles)):
    #     if "jpg" in styles[i]:
    #         iter+=1
    #         print("Name img: " + styles[i])
    #
    #         target_img_path = './results/src.png'
    #         source_img_path = './results/style/' + styles[i]
    #         output_dir = './results/styleout'
    #         temp_x = target_img_path
    #         target_img_path = source_img_path
    #         source_img_path = temp_x
    #
    #         landmarkModel = LandmarkModel(name='landmarks')
    #         landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
    #         face_align(landmarkModel, source_img_path)
    #         face_align(landmarkModel, target_img_path, True, 224)
    #
    #         image_test_auto1(target_img_path, source_img_path, output_dir)

    return iter

@application.route("/hair_style3", methods=['GET', 'POST'])
@cross_origin(origin='localhost:4200')
def hair_style3():
    iter = 0
    print("START HAIR STYLE")
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        file.filename = 'src.png'

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))

        path = (os.path.join(application.config['UPLOAD_FOLDER'], filename))
        print("path :", path)

        result = path.split("/")
        filename2 = result[-1:]
        print("fname :", filename2)
        filename1 = " ".join(filename2)

        print("Done Upload")

    styles = os.listdir("./results/style")

    for i in range (0,len(styles)):
        if "jpg" in styles[i]:
            iter+=1
            print("Name img: " + styles[i])

            target_img_path = './results/src.png'
            source_img_path = './results/style/' + styles[i]
            output_dir = './results/styleout'
            temp_x = target_img_path
            target_img_path = source_img_path
            source_img_path = temp_x

            landmarkModel = LandmarkModel(name='landmarks')
            landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
            face_align(landmarkModel, source_img_path)
            face_align(landmarkModel, target_img_path, True, 224)

            image_test_auto1(target_img_path, source_img_path, output_dir)

    return iter


@application.route("/hair_style", methods=['GET', 'POST'])
@cross_origin(origin='localhost:4200')
def hair_style():
    print("START HAIR STYLE")
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file attached in request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        file.filename = 'src.png'

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
        print(filename)
        file.save(os.path.join(application.config['UPLOAD_FOLDER'], filename))

        path = (os.path.join(application.config['UPLOAD_FOLDER'], filename))
        print("path :", path)

        result = path.split("/")
        filename2 = result[-1:]
        print("fname :", filename2)
        filename1 = " ".join(filename2)

        print("Done Upload")

        # PIC 1

        target_img_path = './ionic-multi-menu/src/assets/images/src.png'
        source_img_path = './ionic-multi-menu/src/assets/images/style/2.jpg'
        output_dir = './ionic-multi-menu/src/assets/images/style'
        temp_x = target_img_path
        target_img_path = source_img_path
        source_img_path = temp_x

        landmarkModel = LandmarkModel(name='landmarks')
        landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
        face_align(landmarkModel, source_img_path)
        face_align(landmarkModel, target_img_path, True, 224)

        image_test_auto1(target_img_path, source_img_path, output_dir)

        # PIC 2

        target_img_path = './ionic-multi-menu/src/assets/images/src.png'
        source_img_path = './ionic-multi-menu/src/assets/images/style/3.jpg'
        temp_x = target_img_path
        target_img_path = source_img_path
        source_img_path = temp_x

        landmarkModel = LandmarkModel(name='landmarks')
        landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
        face_align(landmarkModel, source_img_path)
        face_align(landmarkModel, target_img_path, True, 224)

        image_test_auto1(target_img_path, source_img_path, output_dir)

        # PIC 3

        target_img_path = './ionic-multi-menu/src/assets/images/src.png'
        source_img_path = './ionic-multi-menu/src/assets/images/style/3.jpg'
        temp_x = target_img_path
        target_img_path = source_img_path
        source_img_path = temp_x

        landmarkModel = LandmarkModel(name='landmarks')
        landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
        face_align(landmarkModel, source_img_path)
        face_align(landmarkModel, target_img_path, True, 224)

        image_test_auto1(target_img_path, source_img_path, output_dir)

        # PIC 4

        target_img_path = './ionic-multi-menu/src/assets/images/src.png'
        source_img_path = './ionic-multi-menu/src/assets/images/style/4.jpg'
        temp_x = target_img_path
        target_img_path = source_img_path
        source_img_path = temp_x

        landmarkModel = LandmarkModel(name='landmarks')
        landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
        face_align(landmarkModel, source_img_path)
        face_align(landmarkModel, target_img_path, True, 224)

        image_test_auto1(target_img_path, source_img_path, output_dir)

        # PIC 4

        target_img_path = './ionic-multi-menu/src/assets/images/src.png'
        source_img_path = './ionic-multi-menu/src/assets/images/style/5.jpg'

        temp_x = target_img_path
        target_img_path = source_img_path
        source_img_path = temp_x


        landmarkModel = LandmarkModel(name='landmarks')
        landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
        face_align(landmarkModel, source_img_path)
        face_align(landmarkModel, target_img_path, True, 224)

        image_test_auto1(target_img_path, source_img_path, output_dir)

        return "OK"

@application.route("/get_video", methods=['GET'])
@cross_origin(origin='localhost:4200')
def get_video():
    res_file = './ionic-multi-menu/src/assets/images/dst.mp4'
    return send_file(res_file)

@application.route("/video_swap_file", methods=['GET'])
@cross_origin(origin='localhost:4200')
def video_swap():
    auto_swap_video()

@application.route("/files", methods=['GET', 'POST'])
@cross_origin(origin='localhost:4200')
def get_res_file():
    print("GET FILE")
    res_file = './results/res/dst.png'
    return send_file(res_file, mimetype='image/gif')

@application.route("/src", methods=['GET', 'POST'])
@cross_origin(origin='localhost:4200')
def get_src_img():
    print("GET FILE")
    res_file = './results/src.png'
    return send_file(res_file, mimetype='image/gif')

@application.route("/get_result_video", methods=['GET', 'POST'])
@cross_origin(origin='localhost:4200')
def get_result_video():
    print("GET FILE")
    res_file = './ionic-multi-menu/src/assets/images/demo/output.mp4'
    return send_file(res_file, mimetype='image/gif')

@application.route("/dst", methods=['GET', 'POST'])
@cross_origin(origin='localhost:4200')
def get_dst_img():
    print("GET FILE")
    res_file = './results/dst.png'
    return send_file(res_file, mimetype='image/gif')


if __name__ == '__main__':
    application.debug=True
    application.run(host='0.0.0.0')

    # parser = argparse.ArgumentParser(description="MobileFaceSwap Test")
    # parser.add_argument('--source_img_path', type=str, help='path to the source image')
    # parser.add_argument('--target_img_path', type=str, help='path to the target images')
    # parser.add_argument('--output_dir', type=str, default='results', help='path to the output dirs')
    # parser.add_argument('--image_size', type=int, default=224,help='size of the dltest images (224 SimSwap | 256 FaceShifter)')
    # parser.add_argument('--merge_result', type=bool, default=True, help='output with whole image')
    # parser.add_argument('--need_align', type=bool, default=True, help='need to align the image')
    # parser.add_argument('--use_gpu', type=bool, default=False)
    #
    #
    # args = parser.parse_args()
    # if args.need_align:
    #     landmarkModel = LandmarkModel(name='landmarks')
    #     landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))
    #     face_align(landmarkModel, args.source_img_path)
    #     face_align(landmarkModel, args.target_img_path, args.merge_result, args.image_size)
    # os.makedirs(args.output_dir, exist_ok=True)
    # image_test(args)




