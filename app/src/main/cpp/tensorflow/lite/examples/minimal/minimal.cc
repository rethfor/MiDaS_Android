/* Copyright 2023 by Author EMRE AYTAC. All Rights Reserved.
==============================================================================*/
#include <cstdio>
#include <iostream>
#include <chrono>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"


// Usage: minimal <tflite model>

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "minimal <tflite model>\n");
        return 1;
    }
    const char* filename = argv[1];
    const uint8_t RGB_CHANNEL_COUNT{3};

    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(filename);
    TFLITE_MINIMAL_CHECK(model != nullptr);

    // Build the interpreter with the InterpreterBuilder.
    // Note: all Interpreters should be built with the InterpreterBuilder,
    // which allocates memory for the Interpreter and does various set up
    // tasks so that the Interpreter can read the provided model.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);
    interpreter->SetNumThreads(6);


    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
    /*printf("=== Pre-invoke Interpreter State ===\n");
    tflite::PrintInterpreterState(interpreter.get());*/

    auto input = interpreter->inputs()[0];

    auto input_tensor_batch_size = interpreter->tensor(input)->dims->data[0];
    auto input_tensor_height = interpreter->tensor(input)->dims->data[1];
    auto input_tensor_width = interpreter->tensor(input)->dims->data[2];
    auto input_tensor_channels = interpreter->tensor(input)->dims->data[3];

    const unsigned int total_input_count = input_tensor_width * input_tensor_height;

    std::cout << "The input tensor has the following dimensions: ["
              << input_tensor_batch_size << ","
              << input_tensor_height << ","
              << input_tensor_width << ","
              << input_tensor_channels << "]\n";

    auto output_details = interpreter->output_tensor(0);
    auto output_shape = output_details[0].dims->data[0];
    auto output_height = output_details[0].dims->data[1];
    auto output_width = output_details[0].dims->data[2];

    std::cout << "out shape : " << output_shape << "\n";
    std::cout << "out height : " << output_height << "\n";
    std::cout << "out width : " << output_width << "\n\n";

    cv::Mat image;
    cv::Mat rgb_img;
    cv::Mat input_img;
    cv::Mat output_image;

    double depth_min;
    double depth_max;
    cv::Point minLoc;
    cv::Point maxLoc;

    //image = cv::imread("str.jpg", cv::IMREAD_COLOR);
    cv::VideoCapture cap("sample.mp4");
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file\n";
        return -1;
    }
    //int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH)/4;
    //int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT)/4;
    //cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(frame_width,frame_height));
    cv::Scalar mean (0.485, 0.456, 0.406);
    cv::Scalar std (0.229, 0.224, 0.225);

    while (true) {
        cap >> image;
        if (image.empty()) {
            break;
        }

        cv::cvtColor(image, rgb_img, cv::COLOR_BGR2RGB);

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
                *(input_tensor_data_ptr) = input_img.at<cv::Vec3f>(i).val[j]; //Midas lite model tensor input shape is [1,256,256,3].
                input_tensor_data_ptr++;                                         //So access every pixel within cv::Vec3f then assign to input tensor array
            }
        }

        ///===========================================INFERENCE INVOKE==============================================================
        auto start = std::chrono::steady_clock::now();
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Inference failed" << std::endl;
            return -1;
        }
        auto end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time in milliseconds: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << " ms\n";

        ///===========================================READ OUTPUT BUFFERS============================================================
        // Read output buffers
        // Note: The buffer of the output tensor with index `i` of type T can
        // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
        float* results = interpreter->typed_output_tensor<float>(0);

        ///==========================================CREATE OPENCV MAT FROM OUTPUT===================================================
        cv::Mat output_prediction = cv::Mat(output_height, output_width, CV_32F, results);

        cv::minMaxLoc(output_prediction, &depth_min, &depth_max, &minLoc, &maxLoc);

        ///============================================NORMALIZE AND CALCULATE DEPTH======================================================
        cv::Mat normalized_output = (255 * (output_prediction - depth_min) / (depth_max - depth_min));
        normalized_output.convertTo(normalized_output, CV_8UC3);
        cv::resize(normalized_output, output_image, {image.size().width, image.size().height}, cv::INTER_CUBIC);
        cv::applyColorMap(output_image, output_image, cv::COLORMAP_MAGMA);


        imshow("Frame", output_image);
        //video.write(output_image);

        char c=(char)cv::waitKey(1); //ESC key to exit
        if(c==27)
            break;
    }

    // When everything done, release the video capture object
    cap.release();

    // Closes all the frames
    cv::destroyAllWindows();

    //cv::imshow("win", output_image);
    //cv::waitKey(0);

    return 0;
}
