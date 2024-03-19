"""
    Author: Anastasiia Ilina w20061036, Northumbria University
"""

from ultralytics import YOLO
import time
from PIL import Image
import os

# The class is based on the recommendations and tutorials by Ultralytics https://docs.ultralytics.com/modes/train/#augmentation-settings-and-hyperparameters
class Yolov8ObjectDetectionService:
    def train_Yolov8(self, training_data_yaml, epochs_number = 200):

        start_time = time.time()

        # Load a yolov8 model
        model = YOLO('yolov8n.pt')

        # Train the model
        results = model.train(
            data=training_data_yaml,
            epochs=int(epochs_number), imgsz=[336, 256], rect=True)

        end_time = time.time()

        training_time = end_time - start_time
        print(f"Training time for YOLOv8: {training_time} seconds")

    def validate_yolov8(self, weights_best_model_pt_path):
        # Load a .pt model that was saved after training
        model = YOLO(weights_best_model_pt_path)

        # Validate the model
        metrics = model.val()
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list contains map50-95 of each category
        print(metrics)

    def test_yolov8(self, test_imgs_folder, weights_best_model_pt_path):
        labels = ["landmines"]

        # Load a .pt model that was saved after training
        model = YOLO(weights_best_model_pt_path)

        result_folder = '/yolov8/detection_results_custom_2/'
        os.makedirs(result_folder, exist_ok=True)
        os.makedirs(result_folder + 'images/', exist_ok=True)
        for image_path in os.listdir(test_imgs_folder):
            # Run inference on an image
            results = model(os.path.join(test_imgs_folder, image_path))

            for r in results:
                if len(r.boxes) == 0:
                    print("File has no detectioins: ", image_path)
                for i in range(len(r.boxes)):
                    class_id = int(r.boxes.cls[i].item())
                    confidence = int(100 * float(r.boxes.conf[i].item()))
                    left, bottom, right, top = map(int, r.boxes.xyxy[i].cpu().numpy().tolist())  # Remove [0]

                    output_file_path = os.path.join(result_folder,
                                                    os.path.splitext(os.path.basename(image_path))[0] + ".txt")
                    with open(output_file_path, "w") as f:
                        f.write(f"{labels[class_id]} {confidence} {left} {bottom} {right} {top}\n")

                im_array = r.plot()
                im = Image.fromarray(im_array[..., ::-1])
                image_id = image_path.split('_')[0]
                im.save(result_folder + 'images/' + image_id + '_results' + '.jpg')  # save image

    def get_test_speed(self, test_imgs_folder, weights_best_model_pt_path):
        # To get speed for all images:
        start_time = time.time()
        model = YOLO(weights_best_model_pt_path)
        results = model(test_imgs_folder)

        end_time = time.time()

        testing_time = end_time - start_time
        print('Testing time for YOLOv8: ', testing_time)


    def yolo_detect_test_images(self, test_images_folder):
        labels = ["landmines"]

        # Load a pretrained YOLOv8 Classify model
        model = YOLO('runs/detect/train/weights/best.pt')
        result_folder = 'yolov8_detection_results'
        os.makedirs(result_folder, exist_ok=True)
        os.makedirs(result_folder + '/images/', exist_ok=True)
        for image_path in os.listdir(test_images_folder):
            results = model(os.path.join(test_images_folder, image_path))
            for r in results:
                if len(r.boxes) == 0:
                    print("File has no detectioins: ", image_path)
                for i in range(len(r.boxes)):
                    class_id = int(r.boxes.cls[i].item())
                    confidence = int(100 * float(r.boxes.conf[i].item()))
                    left, bottom, right, top = map(int, r.boxes.xyxy[i].cpu().numpy().tolist())  # Remove [0]
                    output_file_path = os.path.join(result_folder, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
                    with open(output_file_path, "w") as f:
                        f.write(f"{labels[class_id]} {confidence} {left} {bottom} {right} {top}\n")
                im_array = r.plot()
                im = Image.fromarray(im_array[..., ::-1])
                image_id = image_path.split('_')[0]
                im.save(result_folder + '/images/' + image_id + '_results' + '.jpg')
