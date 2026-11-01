# VRC MMPose ステージ1 — InstantHMR
## 有益な点
足首・つま先・踵を含む70関節を、動画をアップロードせずに確認できる。公開ONNXは約81MBで、初回の取得後はブラウザに保存して使い回せる。人物の範囲を手動指定すれば検出器の負荷も省ける。入力映像の再生を推論から独立させ、追いつかない推論の待ち行列を作らない。
## 操作
1. 「モデルを準備」。momolesang/InstantHMRから直接取得し、ハッシュと実推論の数値を検査する。
2. 動画を選択。「映像全体」か、人物をドラッグする「手動ROI」で開始できる。「自動検出」だけ約40MBの検出器が追加される。
3. 「推論を開始」。通常再生は最新フレームを処理。全フレーム解析は選んだfpsの時刻へ順にシークする。
4. 身長・画角・カメラ高・ピッチを入力して距離を校正する。単眼の概算であり、モデルのcam_transとは別の値。
5. JSON / CSV / 計測ログを保存する。足首13/14、足先・踵15〜20。顔5、手42、身体・足23の合計70点。
## 座標と品質
元モデルの3Dはrig-localメートル、X右・Y下・Z前。相対プレビューだけ腰9/10の中点を原点にしてYを反転する。2Dは元動画のpx座標へ戻す。関節信頼度はモデルから出力されないため、score=null。画面内点数は認識精度の点数ではない。
JSONのpredictedCameraPositionは元3D+cam_transというモデル予測。cameraPosition / worldPositionは校正後の骨長・相対Zから別途求める。未校正の絶対距離・床座標はnull。足を床へ強制固定しない。内部行列、レンズ歪み、カメラ姿勢、骨長の誤差は単眼距離に影響する。
## フェーズ
フェーズ1：iPad Air M4等で撮影済みの動画を入力。カメラAPIは使わない。同じトップページをInstantHMR版へ置換し、WebGPU推論・切り抜き・表示・計測・ZIP・ChatGPT Sitesを整える。
フェーズ2：iPad Air M4とHuawei P20 LiteをLAN内WebRTCの撮影端末へ拡張。PC側ブラウザで推論し、時刻・内部/外部校正・複数視点統合を検証する。
フェーズ3：実測してからWaveshare 4MP USBカメラ等へ拡張。1〜4視点とGPU競合を検証する。購入を前提としない。
## 4層の責務
サーバー1：静的配信とPC推論ブラウザ。Node.js workerのHTTPをCodex online sandboxで隔離する将来設計。ブラウザ本体はそのsandbox外。
サーバー2：カメラ・撮影・複数台の接続とWebRTCシグナリング。Codex online sandboxのworkerで制御する将来設計。
サーバー3：時刻同期・遅延・校正・距離・複数視点統合。Codex offline sandboxのworkerへ制限付きIPCで渡す将来設計。
サーバー4：ネイティブ送出。SteamVRへ仮想トラッカーを渡す。
4層は4台のPCを意味しない。ルーターだけで閉域を保証せず、LAN/WAN・ファイアウォール・ICE経路・sandbox境界を実測する。現段階でこれらを実装済みと表示しない。
## 検証の範囲
Actionsで元ONNXの入出力と実動画3フレームのCPU推論を検査する。サイト起動時もCPU参照と数値を照合する。ブラウザを起動するActionsジョブとクラウドブラウザでの視覚確認は行わない。Sites上のPoCで動画を入力し、実機で確認する。RTX 5060 Ti 8GB / 4070 Ti SUPERでの60fps、距離精度、複数視点、SteamVRは実機検証が必要。
## 出典
モデル: https://huggingface.co/momolesang/InstantHMR
コード: https://github.com/Naturalpad-studio/InstantHMR
MMPose: https://github.com/open-mmlab/mmpose
従来の https://huggingface.co/RabbitDaisuke/rtmw3d-webgpu-onnx は変更しない。
