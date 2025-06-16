# Casanovo Non Autoregressive Model 

**Evaluations Folder**

evaluations.ipynb: Has my code for comparing Casanovo outputs with true sequences, creating precision curves, and getting model accuracy.
makeplots.py: Has my code for making loss curves for train/val.
predictions.csv: Example of the way I was getting results, since I was having some issues with the mztab outwriter.


**Notes on model.py**

forward_step: Used for training and for inference. We were talking about also passing constant-length padding during training, but I haven't tried that yet.
predict_step: Currently writing results to a CSV file instead of mztab, which is why we get an error after the first prediction batch.
