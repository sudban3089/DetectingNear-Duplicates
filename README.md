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

Please follow the steps to ensure you first prepare the data to be fed as inputs to the graph neural network, followed by running the node mebedding module and finally the link prediction module

* Data preprocessing and preparation
```bash
PixelandPRNUFeatures_TrainingSet.m
```
```bash
PixelandPRNUFeatures_TestSetNDFI.m
```
```bash
GNNInputs_NDFI.m
```
Run the above scripts in the order mentioned to extract pixel intensity and sensor pattern noise features form the trianing set and the testing set. Then run `GNNInputs_NDFI.m` to prepare the data in the format suitable for GNN. 

* For depth label prediction using GNN
```bash
Nodeembedding.py
```
Run the above script to get the depth label predictions from the graph neural network. Read the script and you can provide `gcn` and `gcn_cheby` to select between ChebNet and GCN. The order of the Chebyshev polynomials can be provded as an input argument. We have used 3rd order Chebsyhev polynomial as it is giving us the best results.   

* For link prediction using sensor pattern noise
```bash
Linkprediction.m
```
Run the above script to construct the IPT using depth labels prodcued by `Nodeembedding.py` and the sensor pattern noise extarcted using `PixelandPRNUFeatures_TestSetNDFI.m`.  

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

