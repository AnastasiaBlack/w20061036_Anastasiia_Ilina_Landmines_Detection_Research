import inspect
import shlex

from utils import GroundTruthConverterXmlToTxt
from utils import NestedFoldersImageRenameService
from detection_metrics_calculator import DetectionMetricsCounter
from utils import CLAHEContrastAugmentationService
from yolov8_object_detection import Yolov8ObjectDetectionService
from ssd_object_detection import SSDObjectDetectionService

xmlConverter = GroundTruthConverterXmlToTxt()
imageRenameService = NestedFoldersImageRenameService()
metricsCounter = DetectionMetricsCounter()
contrastAugmentationService = CLAHEContrastAugmentationService()
yolov8DetectionService = Yolov8ObjectDetectionService()
ssdDetectionService = SSDObjectDetectionService()

functions = {
    "convertXmlToTxt": xmlConverter.convertXmlToTxt,
    "rename_and_copy_images": imageRenameService.rename_and_copy_images,
    "calculate_tp_fp_fn": metricsCounter.calculate_tp_fp_fn,
    "apply_contrast_augmentation": contrastAugmentationService.apply_contrast_augmentation,
    "train_Yolov8": yolov8DetectionService.train_Yolov8,
    "validate_Yolov8": yolov8DetectionService.validate_yolov8,
    "test_Yolov8": yolov8DetectionService.test_yolov8,
    "get_Yolov8_test_speed": yolov8DetectionService.get_test_speed,
    "yolo_detect_test_images": yolov8DetectionService.yolo_detect_test_images,
    "train_ssd": ssdDetectionService.train_ssd,
    "ssd_detect_images": ssdDetectionService.ssd_detect_images
}

while True:
    # Ask for the function name
    func_name = input("Enter one of the functions names "
                      "\n* convertXmlToTxt "
                      "\n* rename_and_copy_images "
                      "\n* calculate_tp_fp_fn "
                      "\n* apply_contrast_augmentation "
                      "\n* train_Yolov8 "
                      "\n* validate_Yolov8 "
                      "\n* test_Yolov8 "
                      "\n* get_Yolov8_test_speed "
                      "\n* yolo_detect_test_images "
                      "\n* train_ssd "
                      "\n* ssd_detect_images "
                      "\n Your input:  ")

    # Check if the function exists
    if func_name in functions:
        # Get the function object from the dictionary
        func = functions[func_name]
        signature = inspect.signature(func)
        expected_params = signature.parameters

        # Asking for the parameters
        params = input(f"Enter the parameters for {func.__name__}: {', '.join(expected_params)}. Note that strings should be provided in quotes: ")
        try:
            # Splitting the parameters and converting them to integers
            params = shlex.split(params)
            print(params)
            # Executing the function with the parameters
            result = func(*params)
            print(f"Result: {result}")
        except (ValueError, TypeError) as e:
            print(f"Error in executing the function: {e}")
    else:
        print("Function not recognized.")

    # Ask if the user wants to continue
    again = input("Do you want to run another function? (yes/no): ")
    if again.lower() != 'yes':
        break