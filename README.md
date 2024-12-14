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
   - This setup allows the use of outdated libraries in a controlled and flexible manner.

2. **Data Preparation**:
   - Custom training data consists of 500 images annotated with bounding boxes using `labelImg` in a Windows environment.
   - Images were resized to **256x256x3** for fast inference on edge devices.
   - Upload the `rockbag_figure.zip` (or a similar dataset archive) to the `content/` directory in Colab.

3. **Training Configuration**:
   - The external `train.py` script is designed for advanced customization. Users can edit it to modify key hyperparameters, including:
     - **Batch size**
     - **Number of epochs**
     - **Backbone architecture**
   - Once the dataset and script are uploaded, the notebook executes the training pipeline and generates the `model.tflite` file.

4. **Outputs**:
   - The trained model (`model.tflite`) is saved and ready for deployment in the Android app.

---

## Android App Setup and Deployment

### Prerequisites

*   The **[Android Studio](https://developer.android.com/studio/index.html)** IDE.
    - Tested on Android Studio Flamingo.
*   A physical Android device with developer mode enabled.

### Deployment Steps

1. Open Android Studio and select **Open an existing Android Studio project**.
2. Navigate to and select the `android/` directory of the project. Click **OK**.
3. If prompted for a Gradle Sync, click **OK** to sync dependencies.
4. Replace the default model in the `assets/` folder with the custom-trained model file (`model.tflite`).
5. Connect your Android device to your computer.
6. Click the green **Run** arrow in Android Studio to build and deploy the app to the device.
7. After launching the app, use the dropdown menu to select the custom model for inference.

### Prebuilt App

If you want to skip the building process, you can download the prebuilt APK with the custom model included from [this Dropbox link](https://www.dropbox.com/scl/fi/dfqe9bbnwysucstnby31k/tflite-example-app.zip?rlkey=briqeuq2i99zk058nv32hpofq&st=5xv6wsex&dl=0).

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
