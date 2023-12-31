#include <jni.h>
#include <string>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include <android/log.h>
#include <android/asset_manager_jni.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "info", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "error", __VA_ARGS__)

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_midaslite_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = cv::getVersionString();
    return env->NewStringUTF(hello.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_midaslite_MainActivity_load_1asset(JNIEnv *env, jclass clazz, jlong mat, jobject assetManager) {
    auto* img = (cv::Mat*)mat;
    if (img->empty()) {
        LOGE("Input image is empty!");
        return;
    }
    const uint8_t RGB_CHANNEL_COUNT{3};
    auto* asset_mgr = AAssetManager_fromJava(env, assetManager);
    auto* asset = AAssetManager_open(asset_mgr, "model.tflite", AASSET_MODE_STREAMING);
    size_t size = AAsset_getLength(asset);
    auto* buf = AAsset_getBuffer(asset);
    if(!buf) {
        LOGE("Buffer is NULL!");
        return;
    }
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromBuffer(
            (const char*)buf, size);

    // Build the interpreter with the InterpreterBuilder.
    // Note: all Interpreters should be built with the InterpreterBuilder,
    // which allocates memory for the Interpreter and does various set up
    // tasks so that the Interpreter can read the provided model.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    interpreter->SetNumThreads(6);
    interpreter->AllocateTensors();

    auto input = interpreter->inputs()[0];

    //auto input_tensor_batch_size = interpreter->tensor(input)->dims->data[0];
    auto input_tensor_height = interpreter->tensor(input)->dims->data[1];
    auto input_tensor_width = interpreter->tensor(input)->dims->data[2];
    //auto input_tensor_channels = interpreter->tensor(input)->dims->data[3];

    const unsigned int total_input_count = input_tensor_width * input_tensor_height;

    auto output_details = interpreter->output_tensor(0);
    //auto output_shape = output_details[0].dims->data[0];
    auto output_height = output_details[0].dims->data[1];
    auto output_width = output_details[0].dims->data[2];

    cv::Mat rgb_img, input_img, output_image;

    double depth_min, depth_max;
    cv::Point minLoc, maxLoc;

    cv::Scalar mean (0.485, 0.456, 0.406);
    cv::Scalar std (0.229, 0.224, 0.225);

    cv::cvtColor(*img, rgb_img, cv::COLOR_BGR2RGB);

    ///===============================RESIZE INPUT IMAGE TO TENSOR SHAPE AND SCALE BETWEEN -1,1==================================
    cv::resize(rgb_img, input_img, cv::Size{input_tensor_width, input_tensor_height}, cv::INTER_CUBIC);
    input_img.convertTo(input_img, CV_32F);

    input_img = ((input_img / 255.0 - mean) / std);

    ///===========================================FILL INPUT BUFFERS==============================================================
    // Note: The buffer of the input tensor with index `i` of type T can
    // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);
    float* input_tensor_data_ptr = interpreter->typed_input_tensor<float>(input);

    for (size_t i{0}; i < (total_input_count) - 1; ++i) {
        for(size_t j{0}; j < RGB_CHANNEL_COUNT; j++) {
            *(input_tensor_data_ptr) = input_img.at<cv::Vec3f>(i).val[j];;//Midas lite model tensor input shape is [1,256,256,3].
            input_tensor_data_ptr++;                                         //So access every pixel within cv::Vec3f then assign to input tensor array
        }
    }

    ///===========================================INFERENCE INVOKE==============================================================
    auto start = std::chrono::steady_clock::now();
    if (interpreter->Invoke() != kTfLiteOk) {
        LOGI("Inference failed");
        return;
    }

    auto end = std::chrono::steady_clock::now();
    LOGI("Elapsed time in milliseconds: %lld ms", std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());

    ///===========================================READ OUTPUT BUFFERS============================================================
    // Read output buffers
    // Note: The buffer of the output tensor with index `i` of type T can
    // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
    auto* results = interpreter->typed_output_tensor<float>(0);

    ///==========================================CREATE OPENCV MAT FROM OUTPUT===================================================
    cv::Mat output_prediction = cv::Mat(output_height, output_width, CV_32FC1, results);

    cv::minMaxLoc(output_prediction, &depth_min, &depth_max, &minLoc, &maxLoc);

    ///============================================NORMALIZE AND CALCULATE DEPTH======================================================
    cv::Mat normalized_output = (255 * (output_prediction - depth_min) / (depth_max - depth_min));
    normalized_output.convertTo(normalized_output, CV_8UC1);
    cv::resize(normalized_output, output_image, {img->size().width, img->size().height}, cv::INTER_CUBIC);
    //cv::applyColorMap(output_image, output_image, cv::COLORMAP_MAGMA);

    *img = output_image.clone();
    AAsset_close(asset);

}