# EfficientDet-Lite2 Object Detection for Autonomous Vehicles: TFLite Model Maker (2024)

### Overview

This project is a proof-of-concept application designed to distinguish between rocks and bags using transfer learning with the **EfficientDet-Lite2** model. The hypothetical use case is for autonomous vehicles to make safer steering decisions by identifying objects in their path.  

- **Intended Behavior Logic**:
  - If a **rock** is detected: the vehicle will swerve to avoid the obstacle.
  - If a **bag** is detected: the vehicle will not swerve, reducing unnecessary evasive actions.  

This approach addresses documented cases where autonomous vehicles struggled to differentiate between rocks and bags.

The custom-trained model has been integrated into an Android app based on the [TensorFlow Lite Examples](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) demo app. The app runs on Android devices and allows switching to the custom model via a dropdown menu.

![Example of app detecting objects in a live scene](media/rockbag-tflite-android.gif)

---

## Training Process

The **EfficientDet-Lite2** model was trained using **TensorFlow Lite Model Maker**, which simplifies the process of training and deploying machine learning models. The library leverages transfer learning to reduce the amount of data and training time required, making it an efficient tool for rapid prototyping.

### Steps to Train the Model:
1. **Setup and Environment**:
   - The `notebook_main.ipynb` file should be uploaded to **Google Colab**.
   - This notebook sets up a custom Python environment using **Miniconda** to work around limitations in Colab's pre-installed libraries and fixed Python version. The custom environment ensures compatibility with older dependencies by:
     - Downloading and installing Miniconda.
     - Creating an isolated environment (`myenv`) with Python 3.9.
     - Updating the conda package manager for smooth operation.
   - This setup allows the use of the deprecated tflite-model-maker library with a CPU runtime.

   **Limitations**:  
   - Training in this current setup is restricted to the CPU, which can be time-consuming for larger datasets or longer training durations. GPU acceleration is not currently supported, potentially due to compatibility issues with the conda environment. This is an area of uncertainty, and contributions to investigate and enable GPU support in Colab are welcome and greatly appreciated. Such improvements would significantly enhance the efficiency of the training process.

   This workaround was first introduced by [wwfish](https://github.com/wwfish/tflite-model-maker-workaround), whose contribution has been instrumental in enabling the use of TensorFlow Lite Model Maker in Colab. His work is greatly appreciated and forms the foundation of this project’s training setup.

2. **Data Preparation**:
   - Custom training data consists of **500 images** annotated with bounding boxes using [LabelImg](https://github.com/heartexlabs/labelImg) in a Windows environment.
   - Annotations were saved in the **PascalVOC format**.
   - Images were resized to **256x256x3** for faster inference on edge devices. However, for improved accuracy, it is recommended to train with images resized to **320x320x3**.
   - Upload the `rockbag_figure.zip` (or a similar dataset archive) to the `content/` directory in Colab.

3. **Training Configuration**:
   - The external `train.py` script is designed for advanced customization. Users can edit it to modify key hyperparameters, including:
     - **Batch size**
     - **Number of epochs**
     - **Backbone architecture**
   - Once the dataset and script are uploaded, the notebook executes the training pipeline and generates the `efficientdet-lite2.tflite` file.

4. **Outputs**:
   - The trained model (`efficientdet-lite2.tflite`) is saved and ready for deployment in the Android app.

---

## Android App Setup and Custom Model Integration

### Overview

By following the [Dropbox link](https://www.dropbox.com/scl/fi/dfqe9bbnwysucstnby31k/tflite-example-app.zip?rlkey=briqeuq2i99zk058nv32hpofq&st=5xv6wsex&dl=0), you can download the Android app files required for this project. The downloaded files include everything needed to run the app as-is or integrate a custom TensorFlow Lite model.

### Setup in Android Studio

1. Download the app files from the link above.
2. Open the `android` folder in **Android Studio** (tested with Android Studio Bumblebee).
3. Users can:
   - Run the app as-is by connecting an Android device in developer mode.
   - Integrate their own custom-trained TensorFlow Lite model into the app (see below for instructions).

### Custom Model Integration

The app comes preloaded with the following TensorFlow Lite models:
- **MobileNet V1**
- **EfficientDet Lite0**
- **EfficientDet Lite1**
- **EfficientDet Lite2** (Custom-trained for rock vs. bag detection)

If you want to replace any of these models with your own custom-trained TensorFlow Lite model:
1. Navigate to the `/assets/` folder inside the app files.
2. Replace one of the existing model files with your custom model.
   - Your custom model file must be named exactly the same as the model you are replacing:
     - `mobilenetv1.tflite`
     - `efficientdet-lite0.tflite`
     - `efficientdet-lite1.tflite`
     - `efficientdet-lite2.tflite`
3. Once the custom model is added, rebuild the app in Android Studio to apply the changes.

---

## Version Tracking

The `version-checks/` folder contains screenshots of the tested versions of tools and software used during development. These include:
- Android Studio version
- Gradle JDK and environment details

---

## Future Work

This project lays the groundwork for exploring object detection in autonomous vehicles. Future enhancements could include:
- Expanding the dataset to cover diverse scenarios (e.g., different lighting conditions).
- Experimenting with alternative architectures for improved accuracy.
- Optimizing the app for faster inference on other edge devices.

---

## Acknowledgments

This project was built using the [TensorFlow Lite Examples Repository](https://github.com/tensorflow/examples) as a foundation, with additional modifications to integrate a custom-trained model. Special thanks to [wwfish](https://github.com/wwfish/tflite-model-maker-workaround) for his invaluable contribution in developing the Colab workaround used in this project.
