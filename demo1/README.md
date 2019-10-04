# OpenVINO Sample - Eye Beam using OpenVINO pre-trained models

## 動かすためにすること

### 1. Model Downloaderで下記4モデルをダウンロード
- face-detection-adas-0001
- head-pose-estimation-adas-0001
- gaze-estimation-adas-0002
- facial-landmarks-35-adas-0002
### 2. gaze3.pyの下記2つの変数に適切なパスを設定
- cpu_ext： cpu_extension.dll の絶対パス（素直にOpenVINOをインストールし、その後のセットアップ作業を行ったのなら、C:\Users\user_name\Documents\Intel\OpenVINO\inference_engine_samples_build\intel64\Release に格納されているはず）
- model_base_path：上記でダウンロードしたモデルが格納されているTransportディレクトリまでの絶対パス
