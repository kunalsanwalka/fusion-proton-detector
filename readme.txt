The codes in this directory are for the fusion proton detector synthetic diagnostic.

In principle, one only needs to interact with the full_detector_workflow.py script which calls functions from the other codes.
The detector_response_from_plasma function in full_detector_workflow.py takes in the following-
1. Location of eqdsk file
2. Detector geometries
3. Radial profiles of basic plasma parameters
And put out the following-
1. Detector response in counts/ms with error bars

This code can then calculate f(v) across (r,z) -> predict the fusion reactivity profile -> predict the detector response (with error bars).

The parallelization is done in 2 places in this code-
1. In the reactivity calculation to get f(v,z) when given f(v,z=0). Each flux tube is parallelized.
2. The response of each detector is calculated in parallel.
