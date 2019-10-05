# OpenVINO Sample - 目からビームが出るサンプル

## 動かすためにすること

### 1. Model Downloaderで下記4モデルをダウンロード
- face-detection-adas-0001
- head-pose-estimation-adas-0001
- gaze-estimation-adas-0002
- facial-landmarks-35-adas-0002

Model Downloaderの使い方は[こちら](https://docs.openvinotoolkit.org/latest/_tools_downloader_README.html)です。

### 2. gaze3.pyの下記2つの変数に適切なパスを設定
- cpu_ext： CPU用拡張ライブラリ絶対パス
    - Windows 10: C:\Users\USER_NAME\Documents\Intel\OpenVINO\inference_engine_samples_build\intel64\Release\cpu_extension.dll
    - Linux: /home/USER_NAME/inference_engine_samples_build/intel64/Release/cpu_extension.so
- model_base_path：上記でダウンロードしたモデルが格納されているTransportディレクトリまでの絶対パス
