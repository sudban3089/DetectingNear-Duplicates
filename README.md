# Detecting Near-Duplicates Using Graph Neural Networks and Sensor Pattern Noise

This is a Tensorflow and MATLAB R2018a implementation of detection of near-duplicate face images using graph neural network and sensor pattern noise. The implementation of the graph neural network is adopted from https://github.com/tkipf/gcn See their paper for details about graph convolutional neural network.
 
## Requirements
* MATLAB R2018a (should run on higher versions also but I have not confirmed it)
* Tensorflow >0.12 (should be compatible with higher versions but check for deprecations) 
* Networkx

## Folder organization

* `Filter` and `Functions` contains C++ compiled files for Sensor Pattern Noise (SPN)- PRNU computation from images
(http://dde.binghamton.edu/download/camera_fingerprint/)

* `Noise Templates Enhanced`, `Noise Templates MLE` and `Noise Templates Phase` contain some of the sensor reference patterns used in this work (Rear sensors
have larger sized reference patterns you can add them for evaluation, change the following scripts if you insert/delete reference patterns: `NCC_Computation_MLE.m`, `NCC_Computation_Enhanced.m`, `NCC_Computation_Phase.m` and `DispSensor.m`)

## Steps to run the scripts

Download the folder on your desktop and run the following scripts: (Please ensure you are in the correct working directory) 

* For PRNU Anonymization
```bash
Demo_Anonymization.m
```
* For PRNU Spoofing
```bash
Demo_Spoofing.m
```
For Spoofing use the same sensor for both test
image (**F**.jpg) and candidate image (**F**.jpg) (Front-Front spoofing,
Rear-Rear spoofing but no cross spoofing such as Front-Rear OR
Rear-Front).

## Helper functions (Read the comments included in individual helper functions for better understanding)

* DispSensor.m : Displays the sensor name as evaluated usign the SPN
classifier 

* getFingerprint\_monochrome.m : Used for computing sensor
reference patterns. Please refer to
http://dde.binghamton.edu/download/camera_fingerprint/ for more details

* NCC\_Computation\_Enhanced.m : Correlates sensor reference patterns
with test noise residuals for Enhanced SPN 

* NCC\_Computation\_MLE.m :
Correlates sensor reference patterns with test noise residuals for MLE
SPN 

* NCC\_Computation\_Phase.m : Correlates sensor reference patterns
with test noise residuals for Phase SPN 

* NoiseExtract\_Basic.m :
Sensor pattern noise extraction used in computing Sensor Reference
pattern for Enhanced SPN and test noise residual for Phase SPN

* NoiseExtractFromImage\_Enhanced, NoiseExtract\_Enhanced.m : Sensor
pattern noise extraction used in computing test noise residual for
Enhanced SPN 

* NoiseExtractFromImage\_MLE, NoiseExtract\_MLE.m : Sensor
pattern noise extraction used in computing Sensor Reference pattern for
MLE SPN and test noise residual for MLE SPN

* NoiseExtractFromImage\_Phase, NoiseExtract\_Phase.m : Sensor pattern
noise extraction used in computing Sensor Reference pattern for Phase
SPN 

* normcor.m : Computes the normalized cross correlation

## Notes

* The scripts provided are generated using training images form the Labeled Faces in the Wild Datset http://vis-www.cs.umass.edu/lfw/ and test images from Near-Duplicate Face Images - Set I http://iprobe.cse.msu.edu/dataset_detail.php?id=1&?title=Near-Duplicate_Face_Images_(NDFI) You can use any other test set

* We used Enhanced PRNU for sensor pattern nosie extraction. See the paper C. T. Li, "Source Camera Identification Using Enhanced Sensor Pattern
Noise," IEEE T-IFS 2010

## References

Please read our previous work to learn more about the face phylogeny:

```
@inproceedings{TBIOM2020_FacePhylo,
  title={Face Phylogeny Tree USing Basis Functions},
  author={Banerjee, Sudipta and Ross, Arun},
  booktitle={IEEE Transactions on Biometrics, Behavior and Identity Science (T-BIOM)},
  volume={2},
  issue={4},
  pages={310-325}.
  year={2020}
}
```

```
@inproceedings{BTAS2019_FacePhylo,
  title={Face Phylogeny Tree: Deducing Relationships Between Near-Duplicate Face Images Using Legendre Polynomials and Radial Basis Functions},
  author={Banerjee, Sudipta and Ross, Arun},
  booktitle={IEEE 10th International Conference on Biometrics Theory, Applications and Systems (BTAS)},
  year={2019}
}
```

