"""
    Author: Anastasiia Ilina w20061036, Northumbria University
"""

import os
import subprocess
import re
import tarfile
import urllib.request
import time
import datetime
import tensorflow as tf
import cv2
import numpy as np
import glob
import random
from tensorflow.lite.python.interpreter import Interpreter

TENSORFLOW_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
DETECTION_CONFIGS_TF_GIT_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/'
GENERAL_SSD_FOLDER = 'SSD'

# The code is based on Tensorflow models available under public Apache License Version 2.0 license on the Github: https://github.com/tensorflow/tensorflow
class SSDObjectDetectionService:
    def get_ssd_model_from_tensorflow_repo(self):
        tf2_setup_path = os.path.join(os.getcwd(), 'models', 'research', 'object_detection', 'packages', 'tf2',
                                      'setup.py')
        setup_path = os.path.join(os.getcwd(), 'models', 'research', 'setup.py')

        # Clone the TensorFlow models repository
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/tensorflow/models"], check=True)

        # Change directory to models/research and compile the Protocol Buffers
        subprocess.run(["protoc", "object_detection/protos/*.proto", "--python_out=."], shell=True,
                       cwd="models/research/", check=True)

        # Modify setup.py file to install the tf-models-official repository targeted at TF v2.8.0
        with open(tf2_setup_path) as f:
            s = f.read()
        with open(setup_path, 'w') as f:
            # Set fine_tune_checkpoint path
            s = re.sub('tf-models-official>=2.5.1',
                       'tf-models-official==2.8.0', s)
            f.write(s)

        # Install local package "research" from the tensorflow github repository
        package_path = "models/research/"
        try:
            subprocess.run(["pip", "install", package_path], check=True)
            print(f"Package installed successfully from {package_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing package from {package_path}: {e}")

        # Install local package "research" from the tensorflow github repository
        model_builder_test_path = "models/research/object_detection/builders/model_builder_tf2_test.py"
        python_executable = subprocess.check_output(["pipenv", "--py"]).strip().decode("utf-8")

        try:
            subprocess.run([python_executable, model_builder_test_path], check=True)
            print(f"Successfully tested model installation by running {model_builder_test_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error running test on model builder from {model_builder_test_path}: {e}")

    def train_ssd(self, train_record_fname, val_record_fname, label_map_pbtxt_fname):
        if not os.path.exists(GENERAL_SSD_FOLDER):
            os.makedirs(GENERAL_SSD_FOLDER)
        # Change directory to "SSD"
        os.chdir(GENERAL_SSD_FOLDER)

        # self.get_ssd_model_from_tensorflow_repo()
        self.create_custom_config(train_record_fname, val_record_fname, label_map_pbtxt_fname)
        self.train()

        # Change directory back to the root
        os.chdir("../")


    def train(self):
        # Set the path to the custom config file and the directory to store training checkpoints in
        pipeline_file = r'models\ssd\pipeline_file.config'
        model_dir = r'ssd\training'
        num_steps = "20"
        python_executable = subprocess.check_output(["pipenv", "--py"]).strip().decode("utf-8")
        command = [
            python_executable, "models/research/object_detection/model_main_tf2.py",
            "--pipeline_config_path", pipeline_file,
            "--model_dir", model_dir,
            "--alsologtostderr",
            "--num_train_steps", str(num_steps),
            "--sample_1_of_n_eval_examples", "1"
        ]

        # Record the start time
        start_time = time.time()

        # Run the command
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during training: {e}")
            return

        # Record the end time
        end_time = time.time()

        # Calculate and print the training duration
        duration = end_time - start_time
        print(f"Training duration: {datetime.timedelta(seconds=duration)}")

        output_directory = 'custom_model_lite'

        # Path to training directory (the conversion script automatically chooses the highest checkpoint file)
        last_model_path = 'ssd/training'

        saveTfLiteModelcommand = [
            python_executable, "models/research/object_detection/export_tflite_graph_tf2.py",
            "--trained_checkpoint_dir", last_model_path,
            "--output_directory", output_directory,
            "--pipeline_config_path", pipeline_file
        ]
        try:
            subprocess.run(saveTfLiteModelcommand, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during training: {e}")
            return

        # Convert exported graph file into TFLite model file
        converter = tf.lite.TFLiteConverter.from_saved_model(
            'custom_model_lite\saved_model')
        tflite_model = converter.convert()

        with open(
                'custom_model_lite\detect.tflite',
                'wb') as f:
            f.write(tflite_model)


    def create_custom_config(self, train_record_fname, val_record_fname, label_map_pbtxt_fname):
        # The chosen_model variable can be changed to deploy different models available in the TF2 object detection zoo
        chosen_model = 'ssd-mobilenet-v2-fpnlite-320'
        MODELS_CONFIG = {
            'ssd-mobilenet-v2': {
                'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
                'base_pipeline_file': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
                'pretrained_checkpoint': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
            },
            'efficientdet-d0': {
                'model_name': 'efficientdet_d0_coco17_tpu-32',
                'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
                'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
            },
            'ssd-mobilenet-v2-fpnlite-320': {
                'model_name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
                'base_pipeline_file': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config',
                'pretrained_checkpoint': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
            },
        }

        model_name = MODELS_CONFIG[chosen_model]['model_name']
        pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
        base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']

        # Create "ssd" folder for holding pre-trained weights and configuration files
        models_directory = 'models/ssd/'
        if not os.path.exists(models_directory):
            os.makedirs(models_directory)

        # Change directory to "ssd"
        os.chdir(models_directory)

        # Download pre-trained model weights
        download_tar_url = TENSORFLOW_MODEL_URL + pretrained_checkpoint
        urllib.request.urlretrieve(download_tar_url, pretrained_checkpoint)

        # Extract the tar file
        with tarfile.open(pretrained_checkpoint) as tar:
            tar.extractall()

        # Download Tensorflow training configuration file for model
        download_config_url = DETECTION_CONFIGS_TF_GIT_URL + base_pipeline_file
        urllib.request.urlretrieve(download_config_url, base_pipeline_file)

        print("Download and extraction complete.")

        # Set training parameters for the model
        num_steps = 20
        batch_size = 16

        # Set file locations and get number of classes for config file
        base_dir = "models/ssd/"
        # Change directory to root
        os.chdir("../../")

        pipeline_fname = os.path.join(base_dir, base_pipeline_file)
        fine_tune_checkpoint = base_dir + model_name + '/checkpoint/ckpt-0'
        from object_detection.utils import label_map_util
        label_map = label_map_util.load_labelmap(label_map_pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        num_classes = len(category_index.keys())
        print('Total classes:', num_classes)

        print('Writing custom configuration file')
        with open(pipeline_fname) as f:
            s = f.read()

        with open(base_dir + 'pipeline_file.config', 'w') as f:
            s = re.sub('fine_tune_checkpoint: ".*?"',f'fine_tune_checkpoint: "{fine_tune_checkpoint}"', s)
            train_record_fname = train_record_fname.replace("\\", "/")
            s = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', f'input_path: "{train_record_fname}"', s)
            val_record_fname = val_record_fname.replace("\\", "/")
            s = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', f'input_path: "{val_record_fname}"', s)
            label_map_pbtxt_fname = label_map_pbtxt_fname.replace("\\", "/")
            s = re.sub('label_map_path: ".*?"', f'label_map_path: "{label_map_pbtxt_fname}"', s)
            s = re.sub('batch_size: [0-9]+', f'batch_size: {batch_size}', s)
            s = re.sub('num_steps: [0-9]+', f'num_steps: {num_steps}', s)
            s = re.sub('num_classes: [0-9]+', f'num_classes: {num_classes}', s)
            s = re.sub('fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "detection"', s)

            if chosen_model == 'ssd-mobilenet-v2':
                s = re.sub('learning_rate_base: .8',
                           'learning_rate_base: .08', s)

                s = re.sub('warmup_learning_rate: 0.13333',
                           'warmup_learning_rate: .026666', s)

            f.write(s)

        print(f"Custom configuration file written to {os.path.join(base_dir, 'pipeline_file.config')}")


    def ssd_detect_images(self, path_to_test_images, path_to_label_txt, num_test_images=118, savepath='ssd_detections', min_conf = 0.5):
        if not os.path.exists(GENERAL_SSD_FOLDER):
            os.makedirs(GENERAL_SSD_FOLDER)
        # Change directory to "SSD"
        os.chdir(GENERAL_SSD_FOLDER)

        min_conf = float(min_conf)
        modelpath = "custom_model_lite/detect.tflite"
        os.makedirs(savepath, exist_ok=True)
        num_test_images = int(num_test_images)
        os.makedirs(savepath + "/images", exist_ok=True)

        # Start counting time for profiling purposes
        start_time = time.time()

        # Grab filenames of all images in test folder
        images = glob.glob(path_to_test_images + '/*.jpg') + glob.glob(path_to_test_images + '/*.JPG') + glob.glob(
            path_to_test_images + '/*.png') + glob.glob(path_to_test_images + '/*.bmp')

        # Load the label map into memory
        with open(path_to_label_txt, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        # Load the Tensorflow Lite model into memory
        interpreter = Interpreter(model_path=modelpath)
        interpreter.allocate_tensors()

        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        float_input = (input_details[0]['dtype'] == np.float32)
        input_mean = 127.5
        input_std = 127.5

        # Randomly select test images
        images_to_test = random.sample(images, num_test_images)

        # Loop over every image and perform detection
        for image_path in images_to_test:
            # Load image and resize to expected shape [1xHxWx3]
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imH, imW, _ = image.shape
            image_resized = cv2.resize(image_rgb, (width, height))
            input_data = np.expand_dims(image_resized, axis=0)
            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if float_input:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[1]['index'])[0]
            classes = interpreter.get_tensor(output_details[3]['index'])[0]
            scores = interpreter.get_tensor(output_details[0]['index'])[0]

            detections = []

            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                if ((scores[i] > min_conf) and (scores[i] <= 1.0)):
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    ymin = int(max(1, (boxes[i][0] * imH)))
                    xmin = int(max(1, (boxes[i][1] * imW)))
                    ymax = int(min(imH, (boxes[i][2] * imH)))
                    xmax = int(min(imW, (boxes[i][3] * imW)))
                    object_name = labels[int(classes[i])]
                    detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

            # Get filenames and paths
            image_fn = os.path.basename(image_path)
            base_fn, ext = os.path.splitext(image_fn)
            txt_result_fn = base_fn + '.txt'
            txt_savepath = os.path.join(savepath, txt_result_fn)

            # Write results to text file
            # (Using format defined by https://github.com/Cartucho/mAP, which will make it easy to calculate mAP)
            with open(txt_savepath, 'w') as f:
                for detection in detections:
                    f.write('%s %.4f %d %d %d %d\n' % (
                        detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))
            for idx, detection in enumerate(detections):
                object_name, confidence, xmin, ymin, xmax, ymax = detection
                # Convert coordinates to integers
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                # Draw rectangle
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                # Add label with object name and confidence score
                label = f"{object_name}: {confidence:.2f}"
                cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # Save the image with detection
                output_path = os.path.join(savepath + "/images/", f"detection_{base_fn}.jpg")
                cv2.imwrite(output_path, image)

        end_time = time.time()
        duration = end_time - start_time
        print("Time for detection of 128 images with SSD is ", duration)

        # Change directory back to the root
        os.chdir("../")
        return
