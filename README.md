# ArecaNutImageProcessing
Developed an image processing system capable of automatically detecting and classifying arecanut as healthy or diseased. The system aims to minimize labor costs and time spent on manual inspection, thereby improving the income of farmers involved in arecanut cultivation.

Collab Link Here: !https://colab.research.google.com/drive/1Wc83uTR0_XVFRQi9yp7Pr38_peDEFLZ1?usp=sharing
1) The target audience for the project (who the users are)?

The target audience comprises millions of people in Asian countries, including India, Myanmar, Bangladesh, Sri Lanka, and others, who use the product or service in their daily lives."

2) What tools/technologies will you use and why?

HARDWARE REQUIREMENTS

• Processor: i3, i5 or more.

• RAM: 4GB and higher.

• Disk Space: 10GB or more.

• High-resolution camera for capturing areca nut images.

• Internet connectivity for data transfer.

3.2 SOFTWARE REQUIREMENTS

• Python Programming Language: Python is a high-level, interpreted programming language that is widely used in machine learning and image processing. It has a large community of developers and a vast ecosystem of libraries and tools that make it ideal for developing complex projects. Python is an open-source language that is maintained by a community of developers. There are numerous libraries and frameworks available in Python, which makes it an attractive choice for software development projects. One of the key advantages of Python is its simplicity and ease of use. It has a clean syntax, which makes it easy to read and write. This means that developers can write code quickly and easily, which can save time and effort in the development process. Additionally, Python has a large standard library, which means that developers do not have to write code from scratch for common tasks.

• TensorFlow, Keras, and Scikit-learning libraries: TensorFlow is a popular open-source machine learning library developed by Google. It provides a set of tools and APIs for building and training machine learning models, including neural networks. Keras is a high-level API that runs on top of TensorFlow and provides an easy-to-use interface for

Detection of Areca Nut Disease using Image Processing

building deep learning models. Scikit-Learn is a machine learning library for Python that provides simple and efficient tools for data mining and data analysis.

• OpenCV library: OpenCV (Open-Source Computer Vision Library) is an open-source computer vision and machine learning software library. It provides a set of tools and algorithms for image processing, computer vision, and machine learning. OpenCV is widely used in the field of computer vision for tasks such as image segmentation, object detection, and image recognition.

• Python IDLE for development: Python IDLE, also known as Integrated Development Environment for Python, is an official Python development environment that comes bundled with the Python distribution. It is a popular choice for beginners and professionals alike who want a simple, yet powerful environment for writing, testing, and debugging Python code.

3) Data Models?

Decision Tree model and Convolution Neural Network used.

4) System architecture and implementation details (if any) Answered In Progress Check 1

5) What is the overall design? What are the core features?

• Image Acquisition: It is the first step of processing which is to collect the images that are relevant to the project.

• Image preprocessing: It is the essential factor used to enhance image data that eliminates unwilling distortions.

• Segmentation: Is a commonly used technique in digital image processing, for the proposed project Otsu method is applied to obtain global image threshold. • Feature Extraction: Is a step in which it divides and reduces a huge collection of raw data into smaller units. GLCM feature extraction is used to perform this step. The roughness, harshness, and smoothness of an image are measured by the exterior portion of an object, which calculates texture. It facilitates the determination of surface and shape.

• Classification: The process of labelling an image from a pre-specified collection of categories is known as image classification. It infers that given an image as input.

6) How do the requirements/design satisfy the needs of your users?

The objective of this project is to classify areca nuts that can potentially cause cancer from those that are safe for consumption. Areca nuts are widely consumed by millions of people in various countries, and this project aims to enable machines to distinguish between the two types.

7) Screen shots of the user interface or a prototype (if any). & Code snippets or Collab project, then complete code segments.

The code segment is attached at the top. The snippets of the Interface is shown at the end.

8) What is the status of the implementation?

The implementation of the project has made significant progress. Here's a summary:

Image Acquisition: The initial step of collecting relevant images for the project has been completed.

Image Preprocessing: This essential step has been implemented to enhance image data, eliminating unwanted distortions.

Segmentation: The project has reached the segmentation phase, where the Otsu method is applied to obtain a global image threshold. This is a common technique in digital image processing.

Feature Extraction: The implementation has progressed to feature extraction, a step where a large collection of raw data is divided and reduced into smaller units. GLCM (Gray Level Co-occurrence Matrix) feature extraction is used for this purpose. It measures the roughness, harshness, and smoothness of an image by calculating texture, facilitating the determination of surface and shape.

Classification: The current status indicates that the final classification is the only step left to complete. Image classification involves labeling an image from a pre-specified collection of categories. It implies determining the category of an image given as input.

Additionally, it's mentioned that a graphical user interface (GUI) has been built using the Python module.

In summary, the project has progressed well, with most of the processing steps implemented, and it seems to be in its final stages with only the classification step remaining. The combination of image acquisition, preprocessing, segmentation, feature extraction, and the development of a GUI demonstrates a comprehensive approach to image processing.

[![ArecaNut](https://img.youtube.com/vi/YourYouTubeVideoID/0.jpg)](https://www.youtube.com/watch?v=YourYouTubeVideoID)


