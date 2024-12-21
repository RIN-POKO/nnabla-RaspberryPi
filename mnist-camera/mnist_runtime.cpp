#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <nbla/singleton_manager.hpp>
#include <nbla_utils/nnp.hpp>
#include <fstream>
#include <sstream>
#include <string>
#include <csignal> // SIGINT用
#include "LCCV/include/lccv.hpp"

using namespace cv;
using namespace std;

#define OFFSET_X (104)
#define OFFSET_Y (0)
#define CLIP_WIDTH (112)
#define CLIP_HEIGHT (224)

#define DNN_WIDTH (28)
#define DNN_HEIGHT (28)

// SIGINTを受け取ったかを管理するフラグ
volatile sig_atomic_t stopFlag = 0;

// SIGINTハンドラ
void handleSigint(int signal)
{
    stopFlag = 1;
}

// Function to process the frame
Mat processFrame(const Mat &frame)
{
    Rect roi(OFFSET_X, OFFSET_Y, CLIP_WIDTH, CLIP_HEIGHT);
    Mat croppedFrame = frame(roi);

    Mat resizedFrame;
    resize(croppedFrame, resizedFrame, Size(DNN_WIDTH, DNN_HEIGHT));

    Mat grayFrame;
    cvtColor(resizedFrame, grayFrame, COLOR_BGR2GRAY);

    return grayFrame;
}

void convertToMnistFormat(const Mat &image, uint8_t *data)
{
    if (image.cols != 28 || image.rows != 28)
        throw runtime_error("Image size must be 28x28.");

    if (image.type() != CV_8U)
        throw runtime_error("Image must be single-channel 8-bit.");

    memcpy(data, image.data, 28 * 28 * sizeof(uint8_t));
}

void read_pgm_mnist(const string &filename, uint8_t *data)
{
    ifstream file(filename, ios::binary);
    string buff;

    getline(file, buff);
    if (buff != "P5")
        throw runtime_error("Only P5 is supported.");

    getline(file, buff);
    while (buff[0] == '#')
        getline(file, buff);

    stringstream ss(buff);
    int width, height;
    ss >> width >> height;
    if (width != 28 || height != 28)
        throw runtime_error("Image size must be 28x28.");

    getline(file, buff);
    int maxval;
    ss.clear();
    ss.str(buff);
    ss >> maxval;
    if (maxval != 255)
        throw runtime_error("maxVal must be 255.");

    file.read(reinterpret_cast<char *>(data), width * height);
    if (!file)
        throw runtime_error("Failed to read image data.");
}

int main(int argc, char *argv[])
{
    if (argc < 2 || argc > 3)
    {
        printf("Usage: %s nnp_file\n", argv[0]);
        return -1;
    }

    const string nnp_file(argv[1]);
    string executor_name = (argc == 4) ? argv[3] : "runtime";

    nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
    nbla::utils::nnp::Nnp nnp(cpu_ctx);

    nnp.add(nnp_file);
    auto executor = nnp.get_executor(executor_name);
    executor->set_batch_size(1);

    auto x = executor->get_data_variables().at(0).variable;
    uint8_t *data = x->variable()->cast_data_and_get_pointer<uint8_t>(cpu_ctx);

    // const string tempFilename = "camera_frame.jpg";

    cv::Mat image;
    lccv::PiCamera cam;
    cam.options->photo_width = 320;
    cam.options->photo_height = 240;
    cam.options->verbose = true;

    while (!stopFlag)
    {
        if (!cam.capturePhoto(image))
        {
            std::cout << "Camera error" << std::endl;
        }

        Mat processedFrame = processFrame(image);

        try
        {
            convertToMnistFormat(processedFrame, data);
        }
        catch (const exception &e)
        {
            printf("Error processing image: %s\n", e.what());
            break;
        }

        // imwrite("processed_frame.pgm", processedFrame);
        // try
        // {
        //     read_pgm_mnist("processed_frame.pgm", data);
        // }
        // catch (const exception &e)
        // {
        //     printf("Error reading PGM: %s\n", e.what());
        //     break;
        // }

        executor->execute();

        auto y = executor->get_output_variables().at(0).variable;
        const float *y_data = y->variable()->get_data_pointer<float>(cpu_ctx);
        int prediction = 0;
        float max_score = -1e10;

        printf("Prediction scores:");
        for (int i = 0; i < 10; ++i)
        {
            if (y_data[i] > max_score)
            {
                prediction = i;
                max_score = y_data[i];
            }
            printf(" %d: %.6f \n", i, y_data[i]);
        }
        printf("Predicted label: %d (score: %.6f)\n", prediction, max_score);
        // if (max_score > 0.7f)
        // {
        //     printf("Predicted label: %d (score: %.6f)\n", prediction, max_score);
        // }
        // else
        // {
        //     printf("Predicted label: None\n");
        // }

        // 100ms待機
        // waitKey(100);
    }

    nbla::SingletonManager::clear();
    cout << "プログラムを終了します。" << endl;
    return 0;
}
