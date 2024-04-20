# Real-world indoor mobility experiments with simulated prosthetic vision (SPV)
This repository contains the code that was used for the data-acquisition and analysis of our SPV mobility study published as:
- de Ruyter van Steveninck, J. , van Gestel, T., Koenders, P., van der Ham, G., Vereecken, F., Güçlü, U., ... & van Wezel, R. (2022). Real-world indoor mobility with simulated prosthetic vision: The benefits and feasibility of contour-based scene simplification at different phosphene resolutions. Journal of vision, 22(2), 1-1. https://doi.org/10.1167/jov.22.2.1

## Experiments 
The experiments can be reproduced through the following steps: 
1. Clone this repository and install depencencies.
2. Optional: Connect arduino (for synchronization with external devices, e.g., camera, motion recording, etc).
3. Connect HTC VIVE Headset with 'direct display mode' disabled (HMD should be recognized as a monitor, and the camera as webcam).
4. Run the data-aquisition script: *python Experiments/run.py*
5. Enter participant ID, etc.
6. A set of windows appear. Drag the large window with the VR view to the external monitor (the VIVE HMD).
7. Run the experiment.

## Analysis
1. Download the data from the donders data repository: https://doi.org/10.34973/ymcn-fe47
2. Clone this repository and run the jupyter notebook:  *Analysis/DataAnalysis_v_21Dec2021.ipynb*

## Follow-up work
For more up-to-date phosphene simulation studies from our lab, please refer to the following repositories:
- [Dynaphos](https://github.com/neuralcodinglab/dynaphos): PyTorch-based simulator with improved biological plausibility.
- [SPVGaze](https://github.com/neuralcodinglab/SPVGazeAnalysis): Gaze-contingent phosphene simulation with mixed reality (HTCVive Pro Eye).  
