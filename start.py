import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
from utils.utils import Logger, mkdir
from utils import label_map_util
import json
from util import FaceAnalysis
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = Logger()

def capture_frames_with_tf(root_dir, output_path, detection_graph, sess, frame_interval=3, scale_rate=0.9):
    results = []

    for filename in os.listdir(root_dir):
        if not filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        video_name = os.path.join(root_dir, filename)
        directoryname = os.path.join(output_path, filename.split('.')[0])
        mkdir(directoryname)
        logger.info('Video_name:{}'.format(video_name))
        cam = cv2.VideoCapture(video_name)
        c = 0

        while True:
            ret, frame = cam.read()
            if not ret:
                logger.warning("ret false")
                break
            if frame is None:
                logger.warning("frame drop")
                break

            frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
            r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if c % frame_interval == 0:
                img_size = np.asarray(frame.shape)[0:2]
                faces, scores, classes, num_detections = detect_face_by_frame(detection_graph, sess, r_g_b_frame)

                if faces.shape[0] > 0:
                    best_face_path = os.path.join(directoryname, f"frame_{c}.jpg")
                    cv2.imwrite(best_face_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    logger.info(f"Saved {best_face_path}")

                    results.append({
                        "video": video_name,
                        "captured_frame": os.path.abspath(best_face_path)
                    })
                    break  # Stop after processing the first detected face

            c += 1

        cam.release()

    with open(os.path.join(output_path, 'frames.json'), 'w') as f:
        json.dump(results, f, indent=4)
    return results

def detect_face_by_frame(detection_graph, sess, image_np):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes), int(np.squeeze(num_detections))

def analyze_faces(frames, age_model, gender_model, race_model, emotion_model):
    analyzed_results = []
    for frame_info in frames:
        img_path = frame_info['captured_frame']
        face = cv2.imread(img_path)
        try:
            age_output = age_model.predict(input=face)
            gender_output = gender_model.predict(input=face)
            race_output = race_model.predict(input=face)
            emotion_output = emotion_model.predict(input=face)
            analyzed_results.append({
                "video": frame_info['video'],
                "captured_frame": img_path,
                "age": age_output,
                "gender": gender_output,
                "race": race_output,
                "emotion": emotion_output
            })
        except Exception as e:
            print(f"Error analyzing {img_path}: {e}")
    return analyzed_results

def save_results(results, output_path):
    with open(os.path.join(output_path, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("root_dir", type=str, help='Path to the data directory containing the videos.')
    parser.add_argument('--output_path', type=str, help='Path to save the results', default='facepics')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    root_dir = args.root_dir
    output_path = args.output_path
    mkdir(output_path)

    PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
    PATH_TO_LABELS = './protos/face_label_map.pbtxt'
    NUM_CLASSES = 2

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    detection_graph = tf.Graph()

    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(graph=detection_graph, config=config) as sess:
            frames = capture_frames_with_tf(root_dir, output_path, detection_graph, sess)

    face_analysis_models = ["age", "gender", "race", "emotion"]
    age_model = FaceAnalysis(model_name=face_analysis_models[0])
    gender_model = FaceAnalysis(model_name=face_analysis_models[1])
    race_model = FaceAnalysis(model_name=face_analysis_models[2])
    emotion_model = FaceAnalysis(model_name=face_analysis_models[-1])

    analyzed_results = analyze_faces(frames, age_model, gender_model, race_model, emotion_model)
    save_results(analyzed_results, output_path)

    for result in analyzed_results:
        print(f"Video: {result['video']}")
        print(f"Captured Frame: {result['captured_frame']}")
        print(f"Age: {result['age']}")
        print(f"Gender: {result['gender']}")
        print(f"Race: {result['race']}")
        print(f"Emotion: {result['emotion']}")
        print('-' * 40)

if __name__ == '__main__':
    main()
