#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <nbla/singleton_manager.hpp>
#include <nbla_utils/nnp.hpp>
#include <fstream>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;

#define OFFSET_X        (104)
#define OFFSET_Y        (0)
#define CLIP_WIDTH      (112)
#define CLIP_HEIGHT     (224)

#define DNN_WIDTH       (28)
#define DNN_HEIGHT      (28)

// Function to process the frame
Mat processFrame(const Mat& frame) {
    Rect roi(OFFSET_X, OFFSET_Y, CLIP_WIDTH, CLIP_HEIGHT);
    Mat croppedFrame = frame(roi);

    Mat resizedFrame;
    resize(croppedFrame, resizedFrame, Size(DNN_WIDTH, DNN_HEIGHT));

    Mat grayFrame;
    cvtColor(resizedFrame, grayFrame, COLOR_BGR2GRAY);

    return grayFrame;
}

void read_pgm_mnist(const string &filename, uint8_t *data) {
    ifstream file(filename, ios::binary);
    string buff;

    getline(file, buff);
    if (buff != "P5") throw runtime_error("Only P5 is supported.");

    getline(file, buff);
    while (buff[0] == '#') getline(file, buff);

    stringstream ss(buff);
    int width, height;
    ss >> width >> height;
    if (width != 28 || height != 28) throw runtime_error("Image size must be 28x28.");

    getline(file, buff);
    int maxval;
    ss.clear();
    ss.str(buff);
    ss >> maxval;
    if (maxval != 255) throw runtime_error("maxVal must be 255.");

    file.read(reinterpret_cast<char*>(data), width * height);
    if (!file) throw runtime_error("Failed to read image data.");
}

int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        cerr << "Usage: " << argv[0] << " nnp_file input_image [executor]" << endl;
        return -1;
    }

    const string nnp_file(argv[1]);
    const string input_image(argv[2]);
    string executor_name = (argc == 4) ? argv[3] : "runtime";

    nbla::Context cpu_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
    nbla::utils::nnp::Nnp nnp(cpu_ctx);

    nnp.add(nnp_file);
    auto executor = nnp.get_executor(executor_name);
    executor->set_batch_size(1);

    auto x = executor->get_data_variables().at(0).variable;
    uint8_t *data = x->variable()->cast_data_and_get_pointer<uint8_t>(cpu_ctx);

    const string tempFilename = "camera_frame.jpg";

    while (true) {
        string command = "libcamera-jpeg -o " + tempFilename + " --width 320 --height 240 --nopreview";
        if (system(command.c_str()) != 0) {
            cerr << "Failed to capture image with libcamera-jpeg" << endl;
            break;
        }

        Mat frame = imread(tempFilename);
        if (frame.empty()) {
            cerr << "Failed to load captured image" << endl;
            break;
        }

        Mat processedFrame = processFrame(frame);
        imwrite("processed_frame.pgm", processedFrame);

        try {
            read_pgm_mnist("processed_frame.pgm", data);
        } catch (const exception &e) {
            cerr << "Error reading PGM: " << e.what() << endl;
            break;
        }

        executor->execute();

        auto y = executor->get_output_variables().at(0).variable;
        const float *y_data = y->variable()->get_data_pointer<float>(cpu_ctx);
        int prediction = 0;
        float max_score = -1e10;

        cout << "Prediction scores:";
        for (int i = 0; i < 10; ++i) {
            if (y_data[i] > max_score) {
                prediction = i;
                max_score = y_data[i];
            }
            cout << " " << y_data[i];
        }
        cout << endl << "Prediction: " << prediction << endl;

        if (waitKey(1000) == 'q') {
            break;
        }
    }

    nbla::SingletonManager::clear();
    return 0;
}

