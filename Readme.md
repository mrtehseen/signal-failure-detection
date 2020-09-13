# Signal Failure Prediction
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

