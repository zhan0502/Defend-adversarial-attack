# Defend adversarial attack

The objective of this project is to defend the adversarial attacks against CNN classifier and infer correct predictions for input images into classes of “artifacts”, “cancer regions”, “normal regions” and “other”. The provided CNN classifier needs to be used as a black-box and it is not allowed to develop alternative models. In this report, I will present the implementation of data pre-processing, classifier training and testing procedure, as well as experimental study.  
My proposed defense approach is transformational-based defense:
(a)	Modify the provided images through image transformation using pixel deflection [1] before inputting them into CNN classifier
(b)	Read out the SoftMax values of CNN to make inferences via a pre-trained long short-term memory (LSTM) classifier. 
