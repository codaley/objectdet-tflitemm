# Bag vs. Rock Detection for Autonomous Vehicles

### Overview

This project is a proof-of-concept application designed to distinguish between bags and rocks using transfer learning with the **EfficientDet-Lite2** model. The intended use case is for autonomous vehicles to make safer steering decisions by identifying objects in their path.  
- **Behavior Logic**:
  - If a **rock** is detected: the vehicle will swerve to avoid the obstacle.
  - If a **bag** is detected: the vehicle will not swerve, reducing unnecessary evasive actions.  
This approach addresses documented cases where autonomous vehicles struggled to differentiate between rocks and bags.  

The custom-trained model has been integrated into an Android app based on the [TensorFlow Lite Examples](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android) repository. The app runs on Android devices and allows switching to the custom model via a dropdown menu.

![Example of app detecting objects in a live scene](placeholder.gif)

---

## Build the Demo Using Android Studio

### Prerequisites

*   The **[Android Studio](https://developer.android.com/studio/index.html)** IDE.
    - Tested on Android Studio Flamingo.
*   A physical Android device running **Android 15** or higher.
    - Developer mode must be enabled.

### Building

1. Open Android Studio. From the Welcome screen, select **Open an existing Android Studio project**.
2. Navigate to and select the `tensorflow-lite/examples/object_detection/android` directory. Click **OK**.
3. If prompted for a Gradle Sync, click **OK**.
4. Replace the default model in the `assets/` folder with the custom-trained model file (`model.tflite`).
5. Connect your Android device to your computer and ensure developer mode is enabled.
6. Click the green **Run** arrow in Android Studio to build and deploy the app.

---

## Training Process

The **EfficientDet-Lite2** model was trained using **TensorFlow Lite Model Maker**, which simplifies the process of training and deploying machine learning models on edge devices. Training details include:
- **Dataset**:
  - 500 custom images labeled with bounding box annotations using `labelImg` on Windows.
  - Images resized to **256x256x3** for fast inference on edge devices.
- **Training Environment**:
  - External training script (`train.py`) was used to fine-tune the backbone model and hyperparameters.

---

## App Deployment

1. **Custom Model Integration**:
   - The trained `model.tflite` file has been added to the app's `assets` folder.
   - Users can select the custom model via a dropdown menu in the app interface.
2. **Download the App**:
   - A prebuilt APK, with the custom model included, can be downloaded from [this Dropbox link](#).

---

## Models Used

The app supports multiple TensorFlow Lite models, including:
- Pretrained models like **EfficientDet Lite0**, **EfficientDet Lite1**, and **EfficientDet Lite2**.
- Custom-trained models like the one in this project for detecting bags and rocks.

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
