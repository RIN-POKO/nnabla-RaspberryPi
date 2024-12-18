# コンパイラとフラグ
CXX := g++
CXXFLAGS := `pkg-config --cflags --libs opencv4`
TARGET := camera_processing
SRC := camera_processing.cpp

# デフォルトターゲット
all: $(TARGET)

# コンパイル
$(TARGET): $(SRC)
	$(CXX) -o $(TARGET) $(SRC) $(CXXFLAGS)

# 実行
run: $(TARGET)
	./$(TARGET)

# クリーンアップ
clean:
	rm -f $(TARGET)

