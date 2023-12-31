# Demo integration of MiDaS in Android

This is an integration of MiDaS (a monocular depth estimator deep learning model) to Android in JNI context (C++ interface of Java). In this project, I tried to show that how
MiDaS or any other TensorFlow Lite model can be loaded through assets, used with TensorFlow Lite C++ API and perform some image processing operations with respect to
inferred values from TensorFlow Lite with OpenCV. Also I tried to demonstrate it in one monolithic function to be more clear, so you can see it in ```invokeMidas()``` native function. 

I wrote C++ implementation based on example usage section in [here](https://www.kaggle.com/models/intel/midas/frameworks/tfLite/variations/v2-1-small-lite/versions/1?tfhub-redirect=true)
