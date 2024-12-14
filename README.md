# TFLite Model Maker (2024): EfficientDet-Lite2 Object Detection for Autonomous Vehicles

### Overview

This project uses **TensorFlow Lite Model Maker** in **Google Colab** to perform transfer learning with an **EfficientDet-Lite2** backbone. The goal is to train a custom object detection model using novel data and deploy it to an Android device for real-time inference. 

The training process involves preparing a dataset of annotated images, configuring the training pipeline, and generating a TensorFlow Lite model (`efficientdet-lite2.tflite`). This model is then integrated into an Android app for deployment. The app is capable of performing real-time object detection and includes options to switch between multiple preloaded models.

The use case demonstrates distinguishing between rocks and bags as a proof-of-concept to assist autonomous vehicles in making safer decisions when encountering objects in their path:
- **If a rock is detected**: The vehicle swerves to avoid the obstacle.
- **If a bag is detected**: The vehicle avoids unnecessary evasive actions.

![Example of app detecting objects in a live scene](media/rockbag-tflite-android.gif)

---

## Training Process

The **EfficientDet-Lite2** model is trained using **TensorFlow Lite Model Maker**, which simplifies training and deploying machine learning models. The library uses transfer learning to reduce the amount of data and training time required, making it an efficient tool for prototyping.

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

A big thanks to [wwfish](https://github.com/wwfish/tflite-model-maker-workaround) for first introducing this workaround in July 2023. His contribution forms the foundation of this project’s training process.

2. **Data Preparation**:
   - Training data consists of **500 images** annotated with bounding boxes using [LabelImg](https://github.com/heartexlabs/labelImg) on Windows.
   - Annotations are saved in the **PascalVOC format**.
   - Images are resized to **256x256x3** for faster inference on edge devices. For improved accuracy, training with images resized to **320x320x3** is recommended.
   - Upload the `rockbag_figure.zip` (or a similar dataset archive) to the `content/` directory in Colab.

3. **Training Configuration**:
   - The `train.py` script allows advanced customization of key hyperparameters, including:
     - **Batch size**
     - **Number of epochs**
     - **Backbone architecture**
   - Once the dataset and script are uploaded, the notebook executes the training process and generates the `efficientdet-lite2.tflite` file.

4. **Outputs**:
   - The `efficientdet-lite2.tflite` file is saved and ready to be used in the Android app.

---

## Android App Setup and Custom Model Integration

### Overview

Follow the [Dropbox link](https://www.dropbox.com/scl/fi/dfqe9bbnwysucstnby31k/tflite-example-app.zip?rlkey=briqeuq2i99zk058nv32hpofq&st=5xv6wsex&dl=0) to download the Android app files. These files include everything needed to run the app or integrate a custom TensorFlow Lite model.

### Setup in Android Studio

1. Download the app files from the link above.
2. Open the `android` folder in **Android Studio** (tested with Android Studio Bumblebee).
3. You can:
   - Run the app as-is by connecting an Android device in developer mode.
   - Replace any of the preloaded models with a custom TensorFlow Lite model (see below).

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

The `version-checks/` folder contains screenshots of the tested versions of tools and software used during development, including:
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

This project uses the [TensorFlow Lite Examples Repository](https://github.com/tensorflow/examples) as its foundation, with modifications to integrate a custom-trained model. Special thanks to [wwfish](https://github.com/wwfish/tflite-model-maker-workaround) for his invaluable contribution to the tflite-model-maker/Colab workaround.
