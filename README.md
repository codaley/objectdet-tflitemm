
# Object Detection with TensorFlow Lite Model Maker

### Overview

This project demonstrates the creation of a custom object detection model using **TensorFlow Lite Model Maker** with an **EfficientDet-Lite2** backbone. Leveraging transfer learning, the model is fine-tuned on a novel dataset to detect two custom classes: rocks and bags. The model is trained in **Google Colab** using a custom Python environment and deployed to a Pixel 7a device for real-time inference.

The pipeline includes dataset preparation, hyperparameter configuration, and generating a TensorFlow Lite model (`efficientdet-lite2.tflite`) suitable for integration into an Android app for live object detection. This project addresses challenges in autonomous systems where misidentification of objects like rocks and bags can compromise safety.

- **Example Logic**:
  - Detecting a **rock** prompts the vehicle to swerve, avoiding potential damage.
  - Detecting a **bag** minimizes unnecessary evasive maneuvers.

![Object Detection App](media/rockbag-tflite-android.gif)

---

## Training Process

The **EfficientDet-Lite2** model is trained using **TensorFlow Lite Model Maker**, a high-level library for efficient prototyping. Transfer learning reduces data and computation requirements, accelerating the training process.

### Steps to Train the Model:

#### 1. Setup and Environment
- Upload the `notebook_main.ipynb` file to **Google Colab**.
- The notebook initializes a custom Python environment with **Miniconda** to address Colab's limitations:
  - Installs Miniconda and sets up an isolated environment (`myenv`) with Python 3.9.
  - Ensures compatibility with the TensorFlow Lite Model Maker library in a CPU runtime.
  
**Note**: Training is CPU-bound, which may be slow for larger datasets. GPU support is currently unavailable due to compatibility constraints. Contributions to enable GPU acceleration are welcome.

#### 2. Data Preparation
- Images are downsampled to **256x256x3** for faster edge-device inference. For improved accuracy, downsampling of original images to **320x320x3** is recommended before annotation.
- Bounding box annotations are created using [LabelImg](https://github.com/heartexlabs/labelImg) in **PascalVOC format**.
- Dataset structure:
  ```
  rockbag_figure.zip
  └── rockbag_figure
      ├── train
      │   ├── IMG_001.jpg
      │   ├── IMG_001.xml
      └── validate
          ├── IMG_101.jpg
          ├── IMG_101.xml
  ```
- Upload the dataset (`rockbag_figure.zip`) and an edited `train.py` script with configured hyperparameters (e.g., batch size, epochs).

#### 3. Training Configuration
- Execute the training pipeline in Colab:
  - COCO metrics are reported, including:
    - **mAP** (mean Average Precision): Detection accuracy.
    - **Precision**: Proportion of correct predictions.
    - **Recall**: Proportion of true objects detected.
  - A TensorFlow Lite model (`efficientdet-lite2.tflite`) is generated post-training.

#### 4. Outputs
- The `efficientdet-lite2.tflite` file is ready for deployment.

---

## Android App Setup and Model Integration

### Setup in Android Studio
1. Download app files from [this Dropbox link](https://www.dropbox.com/scl/fi/dfqe9bbnwysucstnby31k/tflite-example-app.zip?rlkey=briqeuq2i99zk058nv32hpofq&st=5xv6wsex&dl=0).
2. Open the `android` folder in **Android Studio**.
3. Options:
   - Run the app as-is.
   - Replace preloaded models with a custom TensorFlow Lite model.

### Custom Model Integration
1. Navigate to the `/assets/` folder.
2. Replace the existing model file (e.g., efficientdet-lite2.tflite) with your custom-trained model. Ensure the custom model filename matches the original model being replaced.
3. Rebuild the app in Android Studio.

---

## Version Tracking

Screenshots of tested software versions are stored in the `version-checks/` directory, documenting:
- Android Studio version
- Gradle and JDK details

---

## Future Work

Potential extensions include:
- Expanding the dataset to cover diverse conditions (e.g., lighting variations).
- Experimenting with alternative model architectures.
- Optimizing inference for additional edge devices.

---

## Acknowledgments

This project builds on the TensorFlow Lite Model Maker workaround shared by [@tomkuzma](https://github.com/tensorflow/tensorflow/issues/60431#issuecomment-1574781146). The Android app is based on the [TensorFlow Lite Examples Repository](https://github.com/tensorflow/examples).
