**Statement of Purpose**
As part of our 20SU work, we will be developing a machine learning inferencing model to detect stressors and affects with an empatica E4 wristband.  This wristband provides real-time physiological signals from the wearer including blood volume pulse, 3-axix accelerometer, galvanic skin response, and an infrared thermopile.  

To perform our prototyping, we will use an open-access dataset that was made for this project, the Wearable Stress and Affect Detection (WESAD) dataset from Robert Bosch GmbH.  The associated paper and dataset are below.

**Paper Summarization**
[1] Schmidt, Philip, et al. "Introducing wesad, a multimodal dataset for wearable stress and affect detection." Proceedings of the 20th ACM International Conference on Multimodal Interaction. 2018.

This work focuses on the creation and documentation of a standardized, multi-modal, physiological signal dataset that can be used for affective computing.  Their dataset comprises a two sensor dataset: 1) the empatica E4 and 2) a RespiBAN Professional chest-worn device. They defined their experimental design as having two main conditions: amusement and stress, shown below.
![image.png](![image](https://user-images.githubusercontent.com/22334349/112694218-f6da6300-8e4f-11eb-8350-5ffe81b76af2.png)

After the data collection, the subjects completed a PANAS self-survey to assess their positive and negative reactions to the events.  They also used a STAI, then SAM, and finally a SSSQ.  These can be used for personalized algorithm development, but the data collections were used as ground truth for the inferencing.  The data processing pipeline followed the standard routine: pre-processing, segmentation, feature extraction, and classification.  Steps relevant to the E4 follow.
1. Each sample of the sensor signals was segmented using a sliding window, with a shift of 0.25 seconds.
2. ACC - 5 second window
Physiological signals - 60 second window
3. **ACC** - statistical features extracted, for each axis and separately as absolute magnitudes.  Peak frequency for each axis.
**BVP** - heart beats found using peak detection -> Heart rate variability ->  the energy in different frequency bands was computed. The frequency bands used, were the ultra low (ULF: 0.01-0.04 Hz), low (LF: 0.04-0.15 Hz),high (HF: 0.15-0.4 Hz) and ultra high (UHF: 0.4-1.0 Hz) band.
**EDA** - 5 Hz lowpass filter -> compute statistical features -> separate skin conductance level (SCL) (tonic) and skin conductance response (SCR) (phasic) per Choi [16] -> count number of peaks in SCR
**TEMP** - statistical features calculated -> slope (∂_TEMP)

(![image](https://user-images.githubusercontent.com/22334349/112694240-035ebb80-8e50-11eb-95fb-d7e1131be5fe.png)

(![image](https://user-images.githubusercontent.com/22334349/112694254-08bc0600-8e50-11eb-87eb-0dd3b2849642.png)

[WESAD Dataset](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29)

There were two classifiers defined for their analysis: a binary classification and a multi-class classifier.  The binary classifier attempted to classify _non-stressed vs. stressed_, with non-stressed consisting of baseline and amusement and stressed as stress.  The multi-class classifier attempted to discriminate between the three states _baseline vs. amusement vs. stressed_. 

To do this, five machine learning architectures were tested including: Decision Tree (DT), Random For-est (RF), AdaBoost (AB), Linear Discriminant Analysis (LDA), andk-Nearest Neighbour (kNN).  The performance of the models were evaluated using the leave-one-subject-out (LOSO) cross-validation (CV) procedure. Hence, the results indicate how a model would generalise and perform on data of a previously unseen subject.  Additionally, the F1 score is calculated since the events are unbalanced, meaning that the events are not of the same length.

For both classification tasks, 16 different modality combinations are evaluated:
- each of the four modalities of the wrist-based device sepa-rately (ACC, BVP, EDA, and TEMP)
- each of the six modalities of the chest-based device separately(ACC, ECG, EDA, EMG, RESP, and TEMP)
- all modalities of one device (wrist or chest)
- all physiological modalities of one device (same as last entry,but without ACC)
- all modalities from both devices (wrist and chest) together
- all physiological modalities from both devices together (sameas last entry, but without ACC)

Though we only care about the ones that deal with the E4 wristband at this time. 

Overall, the best performance result (in terms of accuracy) oneach of the classification task is:
- 80.34% (three-class problem, using all chest-based physio-logical modalities, AB classifier)
- 93.12% (binary  case,  using  all  chest-based  physiologicalmodalities, LDA classifier)

**Dataset Information**
The double-tap signal pattern was used to manually synchronise the two devices’ raw data. The result is provided in the files SX.pkl, one file per subject. This file is a dictionary, with the following keys:
- ‘subject’: SX, the subject ID
- ‘signal’: includes all the raw data, in two fields:
- ‘chest’: RespiBAN data (all the modalities: ACC, ECG, EDA, EMG, RESP, TEMP)
- ‘wrist’: Empatica E4 data (all the modalities: ACC, BVP, EDA, TEMP)
- ‘label’: ID of the respective study protocol condition, sampled at 700 Hz. The following IDs are provided: 0 = not defined / transient, 1 = baseline, 2 = stress, 3 = amusement, 4 = meditation, 5/6/7 = should be ignored in this dataset

Using [1] as our benchmark, we will develop a stock inferencing model and evaluate its performance using common metrics such as mean average error, F1 score, precision, recall, bias analysis, and variance analysis.  This will not only give us insights into where to improve the model, but also allow us to begin developing an error analysis workflow and toolset that can be reused in future projects.  

We are going to approach this in three steps:
1. Recreate and evaluate stock machine learning models,
2. Export model and connect it into a virtual reality environment, perform real-time environment tuning,
3. Develop experimental protocol to perform data collections, begin developing novel inferencing model.
