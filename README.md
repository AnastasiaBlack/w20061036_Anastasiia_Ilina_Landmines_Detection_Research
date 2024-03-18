# w20061036_Anastasiia_Ilina_Landmines_Detection_Research
This repository contains code used within the research project by Anastasiia Ilina dedicated to the landmines detection using YOLOv8 and SSD models.

### Colab run
For the research project, the code was written and run within Colab environment with additional memory. 
The Colab files can be found in this repository. 
The same code in form of .py classes with 'main.py' class for local python run was rewritten and pushed to this repository as well.

### Local run
Pipfile contains the dependencies with the recommended versions. Python 3.8 and Tensorflow 2.8.0 are proved to be compatible for the current project.

### Project structure
The results of the local run of the YOLO and SSD models can be found in 'ssd_detections' and 'yolov8_detection_results' folders, that contain both images with bounding boxes and .txt files with the coordinates for further comparison with the initial ground truth files.

### Application flow
The application interactively offers a user to choose the function that needs to be called and provides a hint on what parameters are expected to be provided for each function.
The example of functions offered for use input: 
convertXmlToTxt, 
rename_and_copy_images, 
calculate_tp_fp_fn, 
apply_contrast_augmentation,
train_Yolov8,
validate_Yolov8,
test_Yolov8,
get_Yolov8_test_speed,
yolo_detect_test_images,
train_ssd,
ssd_detect_images