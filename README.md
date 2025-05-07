# ebimome
エビを揉むプログラムです。

[動作している様子](https://x.com/riikneeyee/status/1912476723373044207)
## 使い方
1. 両手を映します。手が映ると手の画像が重なって表示されます。
2. 手の上の十字マーカーが画面上の円に入ると、画面左上に上(top)か下(bottom)のどちらに触れたか表示されます。
3. 上下交互に触れると、コンボ数が加算されます。両手を映していないとコンボ数は加算されません。
4. 5コンボ以上になるとエビの画像が表示され、音楽が流れます。コンボ加算ごとに徐々に音量が上がります。
5. 最後のコンボの加算から1秒経過すると、音量が下がっていきます。音量が0になるとコンボがリセットされます。音量が0になる前にコンボを再開すれば、音量が上がっていきます。
6. Qキーを押すと終了します。

## 動かすには
以下をご用意ください。
- Pythonの環境

  こちらはPythonのプログラムです。Pythonの環境をご用意ください。

  [Python公式](https://www.python.org/downloads/)
- MediaPipe

  手のランドマーク検出に使用しています。

  [MediaPipe公式](https://ai.google.dev/edge/mediapipe/solutions/setup_python?hl=ja&_gl=1*uazh95*_up*MQ..*_ga*MjI0NzUwNDIyLjE3NDQ3Nzg5OTY.*_ga_P1DBVKWT6V*MTc0NDc3ODk5Ni4xLjAuMTc0NDc3ODk5Ni4wLjAuMjA0NTQ4NTk0OQ..#developer_environment_setup)

- OpenCV

  カメラのアクセス、カメラ画像の表示に使用しています。

  [OpenCV公式](https://docs.opencv.org/4.11.0/d0/d3d/tutorial_general_install.html)

- Pygame

  音楽を流すために使用しています。

  [Pygame公式](https://www.pygame.org/wiki/GettingStarted)

- カメラ

  PCで使用できるカメラをご用意ください。

- hand_landmarker.taskファイル

  MediaPipeで手のランドマーク検出をする時に使用するトレーニング済みのモデルです。公式ページからダウンロードしてください。
    [MediaPipe公式_モデル](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker?hl=ja&_gl=1*7kvp70*_up*MQ..*_ga*NTI4MzUyMTc5LjE3NDU1MDM5NDg.*_ga_P1DBVKWT6V*MTc0NTUwMzk0OC4xLjAuMTc0NTUwMzk0OC4wLjAuMTUxMzc5ODcxOA..#models)

  MpHandLandmarkerクラス内init関数のmodel_asset_pathで指定しています。

- 音楽ファイル

  流したい音楽を用意してください。
  ComboMusicControllerクラス内init関数の引数music_pathにファイルまでのpathを指定してください。
  
## コツ
- 背景が白いと手の検出が外れる気がします。黒い服などを着て試してみてください。
- 手のひらが下に向いていなくても手を検出します。検出がすぐに外れてしまい難しい場合は、手のひらをしっかり映した状態でやってみてください。
- 上下どちらかの円に触れてから1秒以内に反対に触れるとコンボが加算されます。動かしていると検出が外れてしまいコンボが難しい場合は、1秒以内に反対に触れるようにゆっくりと動かしてみてください。

## 素材
手の画像、エビの画像は作者お手製のお粗末なものです。お手元で試す場合はご自身で用意して入れ替えると、よりリッチで楽しいものになると思います。
音楽は配布を控えています。フリー音源などでお試しください。

ファイル名を同じにして置き換えるか、プログラム内のpathを書き換えてください。

### 画像
ImageRendereクラス内init関数の引数で指定しています。
- right_hand_img_path="right_hand.png"
  
  右手の上に表示されるイラスト
  
- left_hand_img_path="left_hand.png"

  左手の上に表示されるイラスト
  
- ebi_img_path="ebi.png"
  
  エビのイラスト

OpenCVのimreadでサポートされている画像形式で試してください。[OpenCV公式_imread](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html#gacbaa02cffc4ec2422dfa2e24412a99e2)

透過部分は切り抜かれます。

### 音楽
ComboMusicControllerクラス内init関数の引数で指定しています。ご自身で用意したものを使用してください。
- music_path="path_to_music_file"

  ダブルクォーテーション内にpathを書いてください。

PygameでサポートされているのはMP3とOGGのようですが、手元で試した際WAVも読み込めました。
[Pygame公式_music](https://www.pygame.org/docs/ref/music.html)

## リスペクト
[元ネタ様](https://www.nicovideo.jp/watch/sm44019051)および[ご本家様](https://shinycolors.idolmaster.jp/)
