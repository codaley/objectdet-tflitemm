# Object Detection with TFLite Model Maker
### Overview

This project demonstrates how to train a custom object detection model using **TensorFlow Lite Model Maker** with an **EfficientDet-Lite2** backbone. The training process utilizes transfer learning to fine-tune the model on novel data, enabling the detection of two custom classes, rocks and bags. The model is trained in **Google Colab** using a custom Python environment and then deployed to a Pixel 7a device for inference.

The training pipeline involves preparing a dataset of annotated images, configuring hyperparameters, and generating a TensorFlow Lite model (`efficientdet-lite2.tflite`). The resulting model is integrated into an Android app, which supports real-time object detection using the rear-facing camera.

This project was inspired by cases where self-driving cars struggled to differentiate between objects like rocks and bags, leading to unsafe decisions. While not designed to interface directly with autonomous vehicles, it explores solutions to these detection challenges and highlights the potential to improve safety in automated systems.

- **Intended Behavior Logic**:
  - If a **rock** is detected, the vehicle would swerve to avoid the obstacle.
  - If a **bag** is detected, the vehicle would not swerve, reducing unnecessary evasive actions.

![Example of app detecting objects in a live scene](media/rockbag-tflite-android.gif)

---

## Training Process

The custom **EfficientDet-Lite2** model is trained using **TensorFlow Lite Model Maker**, a high-level library that simplifies the training and deployment of machine learning models. The library uses transfer learning to reduce the amount of data and training time required, making it an efficient tool for prototyping.

### Steps to Train the Model:
1. **Setup and Environment**:
   - Upload the `notebook_main.ipynb` file to **Google Colab**.
   - The notebook sets up a custom Python environment using **Miniconda** to address limitations in Colab's pre-installed libraries and fixed Python version. This process involves:
     - Downloading and installing Miniconda.
     - Creating an isolated environment (`myenv`) with Python 3.9.
     - Updating the conda package manager for smooth operation.
   - This setup enables the use of the deprecated tflite-model-maker library with a CPU runtime.

   **Limitations**:  
   - Training with this setup is restricted to the CPU, which can be slow for larger datasets. GPU acceleration is not currently supported, possibly due to compatibility issues with the conda environment. Contributions to enable GPU support in Colab are welcome and appreciated.

2. **Data Preparation**:  
- Training data consists of images resized to **256x256x3** prior to annotation, as bounding box labels are tied to image resolution. This size allows for faster inference on edge devices. For higher accuracy, resizing images to **320x320x3** before annotation is recommended.
   - Annotations are created using [LabelImg](https://github.com/heartexlabs/labelImg) with bounding boxes saved in **PascalVOC format**.  
   - Annotations and images are organized into separate folders for training and validation, following this structure:  

     ```
     rockbag_figure.zip
     └── rockbag_figure
         ├── train
         │   ├── IMG_001.jpg
         │   ├── IMG_001.xml
         │   ├── IMG_002.jpg
         │   └── IMG_002.xml
         └── validate
             ├── IMG_101.jpg
             ├── IMG_101.xml
             ├── IMG_102.jpg
             └── IMG_102.xml
     ```

   - The directory is compressed into a `.zip` file (`rockbag_figure.zip`) for upload. If custom data is used, it must follow this folder structure and annotation format for compatibility with the training pipeline.  
   - The `train.py` script is edited prior to upload to configure key hyperparameters, including:  
     - **Batch size**  
     - **Number of epochs**  
     - **Backbone architecture**  
   - Both the customized `train.py` script and the `rockbag_figure.zip` dataset archive are uploaded to the `content/` directory in Colab.

3. **Training Configuration**:  
   - After uploading the dataset and training script, the notebook initiates the model training pipeline.
   - During training, COCO metrics are printed, including:  
     - **mAP (mean Average Precision)**: Overall detection accuracy across thresholds.  
     - **Precision**: Proportion of correct predictions.  
     - **Recall**: Proportion of actual objects detected.  
   - These metrics validate the model's performance and ensure it meets object detection requirements.  
   - The training process generates a TensorFlow Lite model file (`efficientdet-lite2.tflite`), ready for deployment in the Android app.  


4. **Outputs**:
   - The `efficientdet-lite2.tflite` file is saved and ready to be used in the Android app.

---

## Android App Setup and Custom Model Integration

### Setup in Android Studio

1. Download the app files from the [Dropbox link](https://www.dropbox.com/scl/fi/dfqe9bbnwysucstnby31k/tflite-example-app.zip?rlkey=briqeuq2i99zk058nv32hpofq&st=5xv6wsex&dl=0).
2. Open the `android` folder in **Android Studio** (tested with Android Studio Bumblebee).
3. You can:
   - Run the app as-is by connecting an Android device in developer mode.
   - Replace any of the preloaded models with a custom TensorFlow Lite model (see below) then run.

### Custom Model Integration

The app includes the following preloaded TensorFlow Lite models:
- **MobileNet V1**
- **EfficientDet Lite0**
- **EfficientDet Lite1**
- **EfficientDet Lite2** (Custom-trained for rock vs. bag detection)

To replace any of the models with a custom-trained TensorFlow Lite model:
1. Navigate to the `/assets/` folder in the app files.
2. Replace one of the existing model files with your custom model, ensuring it is named exactly the same as the model being replaced:
   - `mobilenetv1.tflite`
   - `efficientdet-lite0.tflite`
   - `efficientdet-lite1.tflite`
   - `efficientdet-lite2.tflite`
3. Rebuild the app in Android Studio to apply the changes.

---

## Version Tracking

The `version-checks/` folder, located in the project repository, contains screenshots of the tested versions of tools and software used during development, including:  
- Android Studio version  
- Gradle JDK and environment details

---

## Future Work

This project lays the groundwork for exploring object detection in autonomous vehicles. Possible future enhancements include:
- Expanding the dataset to include diverse scenarios (e.g., different lighting conditions).
- Experimenting with alternative architectures for improved accuracy.
- Optimizing inference speeds on other edge devices.

---

## Acknowledgments

This project builds upon the tflite-model-maker workaround originally shared by [@tomkuzma](https://github.com/tensorflow/tensorflow/issues/60431#issuecomment-1574781146) in June 2023. Android implementation is based on the Demo App shared on the [TensorFlow Lite Examples Repository](https://github.com/tensorflow/examples).
