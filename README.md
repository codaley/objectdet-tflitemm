# EfficientDet-Lite2 Object Detection for Autonomous Vehicles: TFLite Model Maker (2024)

### Overview

This project is a proof-of-concept application designed to distinguish between rocks and bags using transfer learning with the **EfficientDet-Lite2** model. The intended use case is for autonomous vehicles to make safer steering decisions by identifying objects in their path.  

- **Intended Behavior Logic**:
  - If a **rock** is detected: the vehicle will swerve to avoid the obstacle.
  - If a **bag** is detected: the vehicle will not swerve, reducing unnecessary evasive actions.  

This approach addresses documented cases where autonomous vehicles struggled to differentiate between rocks and bags.

The custom-trained model has been integrated into an Android app based on the [TensorFlow Lite Examples](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) repository. The app runs on Android devices and allows switching to the custom model via a dropdown menu.

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
   - This setup allows the use of the deprecated tflite-model-maker library in a controlled and flexible manner.

2. **Data Preparation**:
   - Custom training data consists of 500 images annotated with bounding boxes using `labelImg` in a Windows environment.
   - Images were resized to **256x256x3** for fast inference on edge devices.
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

## Android App Setup and Deployment

### Prerequisites

*   An Android device with developer mode enabled.
*   The prebuilt APK file, which can be downloaded from [this Dropbox link](https://www.dropbox.com/scl/fi/dfqe9bbnwysucstnby31k/tflite-example-app.zip?rlkey=briqeuq2i99zk058nv32hpofq&st=5xv6wsex&dl=0).

### Deployment Steps

1. Download the prebuilt APK file from the link above.
2. Transfer the APK file to your Android device.
3. On your Android device:
    - Go to **Settings > Security** and enable the option to install apps from unknown sources (if not already enabled).
    - Locate the APK file using your file explorer app and tap it to install.
4. Launch the app once the installation is complete.

### App Details

This app is a fork of TensorFlow Lite's official object detection demo and includes a custom-trained model for detecting rocks and bags, along with three other preloaded models:

- **MobileNet V1**
- **EfficientDet Lite0**
- **EfficientDet Lite1**
- **EfficientDet Lite2** (Custom-trained for rock vs. bag detection)

When the app runs, you can use the dropdown menu to select one of the four models. The **EfficientDet Lite2** model is preloaded as the custom-trained model designed specifically for detecting rocks and bags.

### Custom Model Integration

If you want to use your own custom-trained TensorFlow Lite model in the app:
1. Your custom model must replace one of the existing models in the `/assets/` folder of the app.
2. Name your model file exactly the same as the model you are replacing:
    - `mobilenet_v1.tflite`
    - `efficientdet-lite0.tflite`
    - `efficientdet-lite1.tflite`
    - `efficientdet-lite2.tflite`
3. Once replaced, rebuild the app to include your custom model.

Note: Replacing models and rebuilding the app requires access to the source code and knowledge of Android app development. If you do not wish to modify the app, you can use the provided APK with the preloaded custom rock vs. bag detection model.



---

## Models Used

The app supports multiple TensorFlow Lite models, including:
- Pretrained models like **EfficientDet Lite0**, **EfficientDet Lite1**, and **EfficientDet Lite2**.
- Custom-trained models like the one in this project for detecting rocks and bags.

Model downloading and placement into the `assets` folder is handled automatically by the app’s Gradle scripts during the build process.

---

## Version Tracking

The `version-checks/` folder contains screenshots of the tested versions of tools and software used during development. These include:
- TensorFlow Lite Model Maker version.
- Android Studio version.
- Python environment details.

---

## Future Work

This project lays the groundwork for exploring object detection in autonomous vehicles. Future enhancements could include:
- Expanding the dataset to cover diverse scenarios (e.g., different lighting conditions).
- Experimenting with alternative architectures for improved accuracy.
- Optimizing the app for faster inference on other edge devices.

---

## Acknowledgments

This project was built using the [TensorFlow Lite Examples Repository](https://github.com/tensorflow/examples) as a foundation, with additional modifications to integrate a custom-trained model.
