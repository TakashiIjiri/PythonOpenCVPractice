# PythonOpenCVPractice

画像処理・コンピュータビジョンの講義に利用するためのコードです．  
講義資料はこちら http://takashiijiri.com/classes/index.html
  
  
**フォルダ構成**  
- *imgs* 
  - サンプル画像が入っています  
   
- *basicoperation* 
  - 画像IO・グラフ描画・フィルタ処理等に関するサンプルコードが入っています
  - chart2D.py - 二次元のグラフを描画
  - chart3D.py - 三次元のグラフを描画
  - chartDots.py - 散布図を描画
  - convolution1.py - 畳み込み演算（cv2利用）
  - convolution2.py - 畳み込み演算（自作）
  - convRgbGray.py - RGB画像の各チャンネルにアクセスする方法
  - EigenVector.py - 2x2行列がおこす2次元空間の変換を可視化する
  - iteratorSample.py - iteratorの練習
  - loadAndSave.py - 画像を読んで書き出し
  - loadAndShowImage.py - 画像を読んで表示
  
- *fourietransform*  
  - フーリエ変換関連  
  - genChartFourie.py - 講義資料用に関数のグラフを描画する
  - FourierSound.py -  音声をフーリエ変換して，周波数空間でフィルタ処理して逆フーリエ変換するサンプル
  - FourierImg.py -  画像をフーリエ変換して，周波数空間でフィルタ処理して逆フーリエ変換するサンプル（ローパス，バンドバス，ハイパス） 
  - FourierPaint.py - 画像をフーリエ変換して，ユーザが塗った領域のみを利用して逆フーリエ変換することで対話的に画像を再構成するツール「 

- *recognition*   
  - 画像認識に関するもの（特徴ベクトル等）  
  - DoG.py - difference of Gaussianを計算  
  - HarrisCorner.py - ハリスコーナー検出  
  - histogram.py - ヒストグラムの計算 
  - hough.py - ハフ変換  
  - panorama.py - パノラマ合成のサンプル
  - SIFT.py  - 2枚の画像からSIFT特徴を検出しその対応付けを行い可視化
  - templatematching.py - テンプレートマッチング
  
- *deconvolution* :  
  - 逆畳み込み
  - KernelGenerator.py : カーネルを生成するコード
  - deconvolution.py : 逆畳み込みを行うコード（点広がり関数は吉と仮定しコード内で生成する．単純な逆数とWeiner filter．）
  





