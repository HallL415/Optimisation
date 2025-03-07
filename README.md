**THIS IS RESEARCH CODE, IT WILL BE CLEANED UP AND RESTRUCTURED FOR READABILITY SOON**

The files TDwLCE (Trotter Decomposition with Linked Cluster Expansion), Functions and params are required to run the code, optionally Forster_FGR may be added for some additional comparisons using Fermi's golden rule (FGR). The params file can be modified to select the system you wish to investigate, allowing for parameter choice such as coupling strength, phonon coupling strengths etc.
Alternatively, params = Parameters() loads the parameters set in params.py with default values and can be updated via params.update()
The dynamics can be computed with Compute_dynamics(), returning the dynamics for either the linear polarisation or population dynamics (up to user choice)
Currently, the Linear Polarisation (LP) for QD-Cavity, QD-QD and QD-QD-Cavity systems may be obtained.
Additionally the Population dynamics can be obtained for a QD-QD system with Foerster-type coupling. 
Also contained in the code is a fitting function to the long time regions of the data. This can be used to extract parameters such as the dephasing rates.
