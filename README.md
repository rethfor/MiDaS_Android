# Demo integration of MiDaS in Android

***Usage:*** Just clone this repository and open the folder in Android Studio as a project then needed things should be automatically configured by IDE. Project configured to work in minimum Android SDK 33 but you can change it from [this line](https://github.com/rethfor/MiDaS_Android/blob/bdbea79faaf08c2cbb6db85334144305f14757c3/app/build.gradle#L11) of ```/app/build.gradle```

This is an integration of MiDaS (a monocular depth estimator deep learning model) to Android in JNI context (C++ interface of Java). In this project, I tried to show that how
MiDaS or any other TensorFlow Lite model can be loaded through assets, used with TensorFlow Lite C++ API and perform some image processing operations with respect to
inferred values from TensorFlow Lite with OpenCV. Also I tried to demonstrate it in one monolithic function to be more clear, so you can see it in ```invokeMidas()``` native function, [here](https://github.com/rethfor/MiDaS_Android/blob/bdbea79faaf08c2cbb6db85334144305f14757c3/app/src/main/cpp/native-lib.cpp#L25). 

* I wrote C++ implementation based on example usage section in [Kaggle's MiDaS model page](https://www.kaggle.com/models/intel/midas/frameworks/tfLite/variations/v2-1-small-lite/versions/1?tfhub-redirect=true)

* I've compiled TensorFlow Lite library with Bazel against Android and it's in ```/app/jniLibs/arm64-v8a```. This project is only available in ```arm64-v8a``` architecture but if you compile TensorFlow Lite for other Android architectures then you can create another directory in ```/app/jniLibs/``` named same as that architecture and put it in there. Also you need to add that architecture to [this line in build.gradle](https://github.com/rethfor/MiDaS_Android/blob/bdbea79faaf08c2cbb6db85334144305f14757c3/app/build.gradle#L19) like ```abiFilters "x86", "armeabi-v7a"```.
  
  
* There is calculation of elapsed time of TensorFlow Lite inference operation in [here](https://github.com/rethfor/MiDaS_Android/blob/bdbea79faaf08c2cbb6db85334144305f14757c3/app/src/main/cpp/native-lib.cpp#L104), you can see it in logs of logcat.
