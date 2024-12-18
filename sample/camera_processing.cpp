#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib> // system()を使うため

using namespace cv;
using namespace std;

#define OFFSET_X        (104)
#define OFFSET_Y        (0)
#define CLIP_WIDTH      (112)
#define CLIP_HEIGHT     (224)

#define DNN_WIDTH       (28)
#define DNN_HEIGHT      (28)

// 画像を加工する関数
Mat processFrame(const Mat& frame) {
    // 1. クリップ
    Rect roi(OFFSET_X, OFFSET_Y, CLIP_WIDTH, CLIP_HEIGHT); // 幅と高さを直接指定
    Mat croppedFrame = frame(roi);

    // 2. リサイズ
    Mat resizedFrame;
    resize(croppedFrame, resizedFrame, Size(DNN_WIDTH, DNN_HEIGHT)); // 28x28にリサイズ

    // 3. モノクロ化
    Mat grayFrame;
    cvtColor(resizedFrame, grayFrame, COLOR_BGR2GRAY);

    return grayFrame;
}

int main() {
    const string tempFilename = "camera_frame.jpg"; // 一時ファイル名
    int frameCount = 0;

    while (true) {
        // libcameraで画像を取得
        string command = "libcamera-jpeg -o " + tempFilename + " --width 320 --height 240 --nopreview";
        if (system(command.c_str()) != 0) {
            cerr << "libcamera-jpegで画像のキャプチャに失敗しました" << endl;
            break;
        }

        // OpenCVで画像を読み込む
        Mat frame = imread(tempFilename);
        if (frame.empty()) {
            cerr << "画像の読み込みに失敗しました" << endl;
            break;
        }

        // フレームを加工
        Mat processedFrame = processFrame(frame);

        // 加工結果を表示
        //imshow("Processed Frame", processedFrame);

        // 加工画像を保存
        string filename = "processed_frame_" + to_string(frameCount) + ".png";
        imwrite(filename, processedFrame);
        cout << "画像を保存しました: " << filename << endl;

        frameCount++;

        // 1秒待つ
        if (waitKey(1000) == 'q') { // 'q'を押すと終了
            break;
        }
    }

    destroyAllWindows(); // OpenCVのウィンドウを破棄
    return 0;
}

