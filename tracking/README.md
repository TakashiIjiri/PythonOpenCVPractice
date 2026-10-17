# Hand Tracking 手法

pip install程度で簡単に導入できるモデルを比較のために動かしてみる

### 全体に利用するライブラリの準備
python -m venv .venv
.venv\Scripts\activate.bat
pip install opencv-python numpy 

### RTMPose-Hand (rtmlib)
必要な項目  
pip install rtmlib  
pip install onnxruntime-gpu  

### Mediapipe 
必要な項目  
pip install mediapipe  





### 必要なモデル
hand_landmarker.task
これを以下の公式配布先からダウンロードし、handtrack_mediapipe.py と同じフォルダに配置
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task


