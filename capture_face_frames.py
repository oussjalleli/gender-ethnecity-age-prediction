import os
import argparse
import json
import cv2
import numpy as np
import tensorflow as tf
from utils.utils import Logger, mkdir
from utils import label_map_util

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = Logger()

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

def crop_and_save_face(image, box, output_filename):
    h, w, _ = image.shape
    ymin, xmin, ymax, xmax = box
    xmin = int(xmin * w)
    xmax = int(xmax * w)
    ymin = int(ymin * h)
    ymax = int(ymax * h)
    
    # Crop the face
    face = image[ymin:ymax, xmin:xmax]
    
    # Resize the cropped face to a standard size (optional)
    face = cv2.resize(face, (224, 224))
    
    # Save the cropped face
    cv2.imwrite(output_filename, face, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    logger.info(f"Saved cropped face to {output_filename}")

def capture_first_frame_with_face(root_dir, output_path, detection_graph, sess, frame_interval=3, scale_rate=0.9):
    for filename in os.listdir(root_dir):
        if not filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        video_name = os.path.join(root_dir, filename)
        logger.info('Processing video: {}'.format(video_name))
        cam = cv2.VideoCapture(video_name)
        c = 0

        while True:
            ret, frame = cam.read()
            if not ret:
                logger.warning("Failed to read frame")
                break
            if frame is None:
                logger.warning("Frame is None")
                break

            frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
            r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if c % frame_interval == 0:
                faces, scores, classes, num_detections = detect_face_by_frame(detection_graph, sess, r_g_b_frame)

                if faces.shape[0] > 0 and scores[0] > 0.5:  # Confidence threshold
                    output_filename = os.path.join(output_path, f"{os.path.splitext(filename)[0]}.jpg")
                    crop_and_save_face(frame, faces[0], output_filename)
                    break  # Stop after processing the first detected face

            c += 1

        cam.release()

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
            capture_first_frame_with_face(root_dir, output_path, detection_graph, sess)

if __name__ == '__main__':
    main()
