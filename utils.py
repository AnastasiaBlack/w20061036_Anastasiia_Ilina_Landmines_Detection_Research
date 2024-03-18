import xml.etree.ElementTree as ET
import cv2
import numpy as np
from glob import glob
import os
import shutil


class GroundTruthConverterXmlToTxt:
    def convertXmlToTxt(self, xml_folder_path, output_path):

        # Path to the output 'txt_metrics' folder
        output_folder_path = os.path.join(output_path, "txt_metrics")
        os.makedirs(output_folder_path, exist_ok=True)

        # Iterate over all XML files in the 'test' folder
        for xml_file_name in os.listdir(xml_folder_path):
            if xml_file_name.endswith(".xml"):
                # Construct the full path to the XML file
                xml_file_path = os.path.join(xml_folder_path, xml_file_name)

                try:
                    # Parse the XML file
                    tree = ET.parse(xml_file_path)
                    root = tree.getroot()

                    # Generate the output file path with the same name as the input XML file but with a ".txt" extension
                    output_file_path = os.path.join(output_folder_path, os.path.splitext(xml_file_name)[0] + ".txt")

                    # Open the output file for writing
                    with open(output_file_path, "w") as f:
                        # Iterate over all objects in the XML file
                        for obj in root.findall(".//object"):
                            class_name = obj.find("name").text
                            xmin = int(obj.find("bndbox/xmin").text)
                            xmax = int(obj.find("bndbox/xmax").text)
                            ymin = int(obj.find("bndbox/ymin").text)
                            ymax = int(obj.find("bndbox/ymax").text)

                            # Write each object to the text file
                            f.write(f"{class_name} {xmin} {ymin} {xmax} {ymax}\n")

                    print(f"Values written to {output_file_path}")

                except ET.ParseError as e:
                    print(f"Error parsing XML file {xml_file_name}: {e}")
                except Exception as e:
                    print(f"An error occurred for XML file {xml_file_name}: {e}")


class NestedFoldersImageRenameService:
    def rename_and_copy_images(self, source_dir, destination_dir):
        # Create the destination directory if it doesn't exist
        os.makedirs(destination_dir, exist_ok=True)

        # Initialize index to avoid overwriting files with the same name
        index = 660

        # Traverse the source directory and its subdirectories
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    # Generate a new name by replacing "Altitude_" with the name of the nearest enclosing folder
                    original_path = os.path.join(root, file)

                    # Extract the name of the nearest enclosing folder
                    nearest_folder = os.path.basename(os.path.normpath(root))

                    # Replace "Altitude_" with "{index}_{nearest_folder}_"
                    new_name = f"{index}_{file.replace('FlightAltitude_', f'{nearest_folder}_')}"
                    new_name = new_name.replace(' ', '_')

                    # Copy the file to the destination directory
                    destination_path = os.path.join(destination_dir, new_name)
                    shutil.copy2(original_path, destination_path)

                    # Increment the index for the next file
                    index += 1

class CLAHEContrastAugmentationService:
    def apply_contrast_augmentation(self, source_folder, destination_folder, clip_limit=1.5, tile_grid_size=(8, 8)):
        """
        Apply CLAHE to all images in a folder and save the transformed images to a destination folder.

        Parameters:
        source_folder (str): Path to the source folder containing the images.
        destination_folder (str): Path to the destination folder where transformed images will be saved.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of the grid for the histogram equalization.
        """
        # Create the destination folder if it does not exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Create a CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # Get a list of all image files in the source folder
        image_files = glob(os.path.join(source_folder, '*.jpg'))

        for image_path in image_files:
            # Read the image in grayscale
            image_path = image_path.encode('utf-8').decode('utf-8')

            # Read the image file into a byte array
            with open(image_path, 'rb') as file:
                image_data = file.read()

            # Convert the byte array to a numpy array
            image_array = np.frombuffer(image_data, dtype=np.uint8)

            # Decode the image from the numpy array
            image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

            # Apply CLAHE to the grayscale image
            clahe_image = clahe.apply(image)

            # Construct the path for the transformed image to be saved
            base_name = os.path.basename(image_path)
            save_path = os.path.join(destination_folder, base_name)
            # Encoding and Decoding the save path
            save_path = save_path.encode('utf-8', 'surrogateescape').decode('utf-8', 'surrogateescape')

            # Save the transformed image
            cv2.imwrite(save_path, clahe_image)
            print(f"Processed and saved: {save_path}")
