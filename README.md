# Codes for the Manuscript

![version](https://img.shields.io/badge/version-1.0-blue)
![Python](https://img.shields.io/badge/Python-3.6-green)
![license](https://img.shields.io/badge/license-MIT-orange)


<p align="center">
  <img src="harmoni_logo.png"/>
</p>

<h2>Harmoni: a Novel Method for Eliminating Spurious Neuronal Interactions due to the Harmonic Components in Neuronal Data</h2>
Mina Jamshidi Idaji, Jaunli Zhang, Tilman Stephani, Guido Nolte, Klaus-Robert Mueller, Arno Villringer, Vadim V. Nikulin
INSERT THE DOIs

(c) Mina Jamshidi (minajamshidi91@gmail.com) @ Neurolgy Dept, MPI CBS, 2021
https://github.com/minajamshidi
(c) please cite the above paper in case of using this code for your research
-----------------------------------------------------------------------

<h3>Abstract</h3>

Cross-frequency synchronization (CFS) has been proposed as a mechanism for integrating spatially and spectrally distributed information in the brain. However, investigating CFS in Magneto- and Electroencephalography (MEG/EEG) is hampered by the presence of spurious neuronal interactions due to non-sinusoidal waveshape of brain oscillations. Such waveshape gives rise to the presence of oscillatory harmonics mimicking genuine neuronal oscillations. Until recently, however, there has been no methodology for removing these harmonics from neuronal data. 

In order to address this long-standing challenge, we introduce a novel method (called HARMOnic miNImization - Harmoni) that removes the signal components which can be harmonics of a non-sinusoidal signal. Harmoni’s working principle is based on the presence of CFS between harmonic components and the fundamental component of a non-sinusoidal signal. 

We extensively tested Harmoni in realistic EEG simulations. The simulated couplings between the source signals represented genuine and spurious CFS and within-frequency phase synchronization. Using diverse evaluation criteria, including ROC analyses, we showed that the within- and cross-frequency spurious interactions are suppressed significantly, while the genuine activities are not affected. Additionally, we applied Harmoni to real resting-state EEG data revealing intricate remote connectivity patterns which are usually masked by the spurious connections. Given the ubiquity of non-sinusoidal neuronal oscillations in electrophysiological recordings, Harmoni is expected to facilitate novel insights into genuine neuronal interactions in various research fields, and can also serve as a steppingstone towards the development of further signal processing methods aiming at refining within- and cross-frequency synchronization in electrophysiological recordings.
-----------------------------------------------------------------------

<h3>Codes and the manuscript figures</h3>

Here, you can see how the scrpts correspond to the manuscript figures:

* figure 2: sawtooth_toy.py

* figure 3: harmoni_blockdiagram.py 

* figure 8: simulations_toys.py 

* figure 9: proof_of_concept.py 

* figure 10: realisticsimulation_results.py 

* figure 11: lemon_nonsin_source_exp.py 

* figures 12, 13, 14: lemon_data_analysis.py 

additionally:

* computing the connectivity of LEMON data: lemon_conn_bandpass.py 

-----------------------------------------------------------------------

<h3>Versions</h3>

###### version 1.0
`Oct. 2021` Codes contributing to the first bioRxiv preprint at xxDOIxx


