# Signal Failure Prediction
[![Deon badge](https://img.shields.io/badge/ethics%20checklist-deon-brightgreen.svg?style=popout-square)](http://deon.drivendata.org/)

 Cellular link failures cause poor user experience perceived as call drop or slowness/stall in data transfers. In this, I investigated a way to generate a model based on MLP (Multi-layer Perceptron) machine learning approach that can be used on Android Devices with the potential features that can predict link loss to a certain efficiency.
 
 
 ## Data
 Features are chosen to represent wireless link performance and model learning data gathered by different Android devices using custom made Android app for collecting signal data for few weeks. They are stored in the csv format and each colums.
 
 Following are the features used in the dataset.
 
### Signal Strength – GSM & 3G/HSPA (RSSI)
 Applicable to GSM and 3G networks. The exact signal strength, often called Received Signal Strength Indication (RSSI), is measured in dBm. The dBm scale is roughly between -50 and -120dBm, with -50 being perfect signal and -120 being when you fall off the network [4]. RSSI measures both the usable signal and the noise in a single figure.
 -50 to -75 dBm – High Signal
 -76 to -90 dBm – Medium Signal
 -91 to -100 dBm – Low Signal
 -101db to -120 dBm – Poor Signal
 
### Signal Strength – 4G/LTE (RSRP)
 LTE signal strength is measured on a different scale than 3G/HSPA, it is measured in Reference Signal Received Power (RSRP). This often ends up being around -20dBm lower than RSSI, so - 100dBm (RSSI) would equate to around -120dbm (RSRP).
 RSCP = RSSI + Ec/Io
 RSRP does a much more accurate job of measuring signal strength than RSSI, as it excludes noise and interference on the network, measuring just the usable portion of the signal. Just because RSRP signals appear lower, it will not mean your signal is worse.
 -75dBm and -88dBm is a strong signal
 -89dBm and -96dBm is a very good signal
 -97dBm and -105dBm is good
 -106dBm and -112dBm is fair
 -113dBm and -125dBm is poor

### SNR
 The “Signal to Noise Ratio” is a measurement that compares the signal strength to the level of background noise. The higher the SNR the better your signal quality will be. The SNR reading will be automatically calculated by the base station in dB. On the SNR scale 4 is poor and 25 is great.
### ASU
 "Arbitrary Strength Unit" is a value that is proportional to your RSRP. The higher the number the better your signal quality. ASU = RSRP + 140, so if you have a RSRP of -100dBm, then the ASU will be 40.
### Ec/Io
 Very similar to SNR above, this measurement is the ratio of signal to interference. With the best being around -10 and the worst quality being around -40.
### RSRQ
 Reference Signal Received Quality is the ratio of usable signal to noise and interference measured in dBm. 
 RSRQ=RSRP-RSSI.
 
 > Features.
![](https://i.ibb.co/fvnMMqB/features.png)


#### The last column corresponds if the link failure occured or not, making it a classification prroblem.

#### The variables included in the data set are receive power, receive signal to noise ratio, transmit power, link error rate, modulation scheme and time of the measurement, the received signal strength (RSS) in dBm, the carrier to interference (C / I) ratio measured in dB and at last the label data whether a signal failure occurred or not as (y=1 | y=0) where 1 defines cellular link failure while 0 describes normal signal parameters. These parameters are selected as a features for determining probable link failure as these signal values describes and withholds and associates with the reason when a signal failure occurs.

## Model Selection

 As Deep CNNs require intensive training, networks such as SSD, R-CNN and YOLO were evaluated as the detection frameworks for the CNN to be extended and fine-tuned, and MobileNet, Inception and ResNet were evaluated as base networks. 
The selection of the base model for the object detection algorithms amongst these state-of-the-art models depends heavily on the speed versus performance payoff between the models. The speed and accuracy varies heavily between the different models as well as the map between the networks. According to this data Tiny YOLO is faster than object detection models built with SSD, but is inferior in detection performance, which is also the case between Faster R-CNN and SSD, where SSD is the faster but lower performing networking.
 In a mobile environment, the FPS that the mobile device is able to perform computations in is crucial for the mobile experience. With a too complex and computationally expensive network, the device would struggle to run the application on more than 1 FPS. According to professional developers at Bontouch, a minimum around 2 FPS is required for a smooth mobile experience, which limited the base network to either MLP or CNN as Faster R-CNN has inferior speed on mobile devices. As for this training so we MLP for the training of the as it is known as sequential approach in keras library of TensorFlow.
In TensorFlow, it is also known as Sequential estimator which determines the binary classification in Deep Learning where different perceptron forms to be a output function.

### Tensorflow-Keras (Multi-Layer-Perceptron)
 The model is trained on TensorFlow keras library which readily helps to easily compile a model on MLP where it specifies every detail of its execution. The Sequential model is a linear stack of layers.
You create a Sequential model (MLP) by passing a list of layer instances to the constructor:
The model needs to know what input shape it should expect. For this reason, the first layer in a Sequential model (and only the first, because following layers can do automatic shape inference) needs to receive information about its input shape.
Before training a model, you need to configure the learning process, which is done via compile method. TensorFlow Keras models are trained on Numpy arrays of input data and labels. For training a model, you will typically use the fit function the compile method.

#### The language used to develop TF model is Python (3.6+) and TensorFlow (1.19+) on Windows OS with JetBrains PyCharm Community Edition 2018.3.4 x64 as a IDE.

## Model Evaluation
 The model is evaluated on the test data split in the data processing phase where each data is chopped down to test the model and get its accuracy on the shelf. The mAP value served as a direct indicator of detection performance in terms of both class prediction and bounding box prediction, and as the base metric for evaluating model performance. As mAP is a commonly used metric, there are multiple open-source libraries and software packages available to evaluate the network mAP performance and TensorFlow has built-in support for calculating the mAP metric via TensorBoard during training and evaluation. By performing a series of experiments, enough data was gathered to evaluate the performance of the models via accuracy and loss which is predefined in the TensorFlow keras library.
The model showed the average accuracy of 0.94 on the test data and average loss of 0.233 as described below:

 > Accuracy.
![](https://i.ibb.co/8mcr2vh/model-accuracy.png)

## Model Saving
 The model will is saved in .H5 format after training so it can be used later on the implementation on different devices, the model saves the weights and variables for operations to deduct inference on the different devices.
 
 [![Twitter](https://img.shields.io/twitter/follow/thetehseen.svg?style=social&label=Follow)](https://twitter.com/thetehseen/)
![GitHub followers](https://img.shields.io/github/followers/mrtehseen?label=GitHub&style=social)

