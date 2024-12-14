# objectdet-tflitemm
Train an EfficientDet model for object detection optimized for mobile inference on Android. This 2024 approach provides a workaround for training using the recently deprecated TFLite Model Maker library on a Google Colab CPU runtime.

Bag vs. Rock Detection for Autonomous Vehicles
This project leverages transfer learning with the EfficientDet-Lite2 model to detect bags and rocks, providing a proof-of-concept solution for autonomous vehicle obstacle detection. The intended use case is to assist in differentiating between rocks and bags in a vehicle's path, a critical challenge as documented cases have shown autonomous vehicles struggling to distinguish between these objects.

Behavior Logic
Rock Detected: The vehicle's steering logic will swerve to avoid a potential hazard.
Bag Detected: The vehicle's steering logic will not swerve, reducing unnecessary evasive actions.

Project Overview
Model Training:

Utilized transfer learning with the TensorFlow Lite Model Maker library.
Novel training data of 500 images was manually labeled using the labelImg tool in a Windows environment.
Images were downsized to 256x256x3 for efficient inference on edge devices like mobile phones.
The train.py script allows fine-tuning of the backbone model and hyperparameters, providing flexibility for advanced users.
Deployment:

The app used for deployment is adapted directly from the TensorFlow Lite Examples Repository.
The custom-trained model file (model.tflite) is copied into the assets folder of the app.
Upon running the app, a dropdown menu allows switching to the newly trained model.
Test Environment:

The trained model was tested on a Google Pixel 7a running Android 15.
Quick inference times were achieved due to optimized image dimensions and model architecture.
Version Tracking:

The version-checks/ folder contains screenshots of the tested software versions used during development to ensure reproducibility.

Use Case and Relevance
This project serves as a foundational step for researchers and developers aiming to explore real-world object detection challenges with TensorFlow Lite. Autonomous vehicle logic must accurately distinguish between objects like rocks and bags to make appropriate decisions, as the consequences of misidentification can impact safety and reliability.

About TensorFlow Lite Model Maker
The TensorFlow Lite Model Maker library simplifies training and deploying ML models on edge devices. It abstracts away many complexities but still allows customization of the backbone model and hyperparameters via external scripts such as train.py.

How to Use the App
Download the App:
The prebuilt Android app, preloaded with the custom-trained model, can be downloaded from this Dropbox link.

Run the App:

Install the app on your Android device.
Open the app and select the custom model from the dropdown menu.
Point the device camera at objects (e.g., rocks and bags) to see real-time predictions.
Model Integration:

The custom model (model.tflite) is already loaded into the app's assets folder.

Folder Structure
assets/: Contains the custom-trained model file.
notebooks/: Training notebooks showcasing the data preprocessing, training, and evaluation process.
train.py: External script for advanced tuning of the model and hyperparameters.
version-checks/: Screenshots of software versions used during development.
data/: Includes training, validation, and test datasets (not included in the repository due to size constraints).

Acknowledgments
This project was built using the TensorFlow Lite Examples repository as a base, with additional modifications for custom model integration. Special thanks to the TensorFlow Lite team for providing a robust library that simplifies the model training and deployment process.

Future Work
This proof-of-concept project can be extended by:

Expanding the dataset to include more diverse scenarios (e.g., different lighting or weather conditions).
Improving model accuracy by experimenting with alternative backbone architectures.
Optimizing inference speeds for other edge devices.

Feel free to explore the repository and adapt it for your own use cases!
