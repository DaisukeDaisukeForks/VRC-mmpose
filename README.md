# VRC MMPose — InstantHMR / Phase 01
足首・つま先・踵を含む70関節を、撮影済み動画からブラウザ内で推論する。主画面はInstantHMR。約81MBの公開ONNXを直接取得・検証・保存し、次回から再利用できる。全体・手動ROIなら人物検出器の追加ダウンロードと推論を省ける。動画表示は姿勢推論から独立し、処理が追いつかないときも古いフレームをキューへ積まない。
公開先: https://vrc-mmpose-video-lab.daisukedaisuke389137.chatgpt.site
原モデル: https://huggingface.co/momolesang/InstantHMR
実装・座標仕様: https://github.com/Naturalpad-studio/InstantHMR
関連する公式MMPose: https://github.com/open-mmlab/mmpose
既存の https://huggingface.co/RabbitDaisuke/rtmw3d-webgpu-onnx は変更しない。Sitesの既存比較ページも変更せず、トップページだけを置換する。GitHubに既存のない比較ページを今回追加しない。
## 使い方
1. PCのChrome / Edgeでトップページを開き、「モデルを準備」。初回はInstantHMR 80,961,622 bytesと推論エンジンを公開フェッチする。SHA-256確認、セッション作成、実推論、CPU参照値との数値照合まで完了してから準備済みにする。
2. iPad Air等で撮影した動画を選択する。サンプル動画も同梱。カメラ入力は実装しない。
3. 初期値は「映像全体」。手動ROIでは映像上をドラッグする。「自動検出」だけRTMDet-s約40MBを追加取得し、WASMで人物検出、関節からの範囲更新と定期再検出を行う。
4. 「推論を開始」。通常再生は処理可能な最新フレームを順次推論する。全フレーム解析は指定fpsの時刻へシークして順次解析する。指定サンプリングfpsと元動画fpsは別である。
5. 身長・画角、またはfx/fy/cx/cyとカメラ高・ピッチを入力すると、骨長と相対Zから単眼の距離・床座標を概算する。モデルの`cam_trans`はこの校正結果と分けて記録する。
6. JSON / CSVで70関節の名前・元動画px座標・相対3D・モデル予測カメラ座標・校正座標を保存する。JSONにはMHR 204パラメーターとshape 45パラメーターも含む。計測ログは実効fps、推論と全体のp50/p95、初回照合結果を保存する。
## 実行
Node.js 22以降。アプリのローカル配信に追加npm依存・ビルドは不要。
```sh
node serve.mjs
```
http://127.0.0.1:4173 を開く。`dist/`が静的配信対象。モデル変換・実ONNX検証・数値テストはGitHub Actionsで行い、作業環境のCPUではモデルをビルド・推論しない。
## 固定したモデル契約
取得元は`momolesang/InstantHMR`、revision `3504446fc31e7f76fdb1cd7e463189e7cf0fdd0f`、ファイル`instanthmr.onnx`。SHA-256は`f717558094f57d7c9cd084d7981874dbcc3f08e7308501eb98c12844e2cce3b7`。モデル本体をこのGitリポジトリやZIPへ同梱・ミラーしない。
Actionsで実グラフを検査した結果、opset17、FP16初期化子826個、外部入出力はFP32、CLIFF角度形式を指定するmetadataなし。現在の学習リポジトリの別モデルをこの公開済みファイルと混同しない。
|入力・出力|形状|意味|
|---|---|---|
|image|1×3×224×224|RGB / ImageNet正規化 / NCHW|
|cliff_cond|1×3|元BBox中心とスケール。余白を加える前に計算|
|mhr_params|1×204|MHR回転パラメーター|
|shape_params|1×45|形状パラメーター|
|cam_trans|1×3|モデル予測カメラ平行移動、m|
|joints_2d|1×70×2|cropの正規化座標[-1,1]|
|joints_3d|1×70×3|rig-local座標、m、X右・Y下・Z前|
人物の長辺×1.2の正方形を黒余白付きで切り抜く。CLIFFは`[(x1+x2)/W-1, (y1+y2)/H-1, max(bw,bh)/max(W,H)]`。元動画から直接224px cropを作り、最大辺1280pxのプレビュー画像をモデル入力には使わない。
MHR70の足首は13/14、左足先・踵は15/16/17、右は18/19/20。腰は9/10、右手首41、左手首62。顔5点、手42点、身体・足23点。表示だけ腰中心・Y上に変換し、`relativeModel`は元のrig-local出力を保つ。COCO133の番号・SimCC復号をInstantHMRへ流用しない。
関節ごとの信頼度はこのモデルの出力にない。画面は有限値かつ画面内の点数を表示し、JSONのscoreはnull、CSVは空欄、`confidenceAvailable=false`。回帰結果が有限であることは、遮蔽が解決したことや精度の証明ではない。
## WebGPUと計測
`onnxruntime-web@1.27.0`のJSEP bundleへユーザー提供のFP32累積パッチを適用する。元JS・glue・WASMとパッチ済みJSをSHA-256で照合し、エンジンもキャッシュする。FP16を理由に速度向上を断定しない。
起動時にActionsで計算した固定入力の2D・3D・カメラ出力と照合する。NHWCが初期化または数値検査に失敗したらNCHWでも検査する。これは特定入力の破綻検出であり、全画像のGPU等価性・精度を保証するものではない。CPU実行はユーザーが選ぶWASM診断モードに限る。必要な非GPU演算がORT内部で分割される可能性はあり、GPU100%実行とは表示しない。
ウォームアップはfps集計から除外。通常再生の実効fpsは処理フレーム数÷経過時間。出力読み戻しを含む推論時間、検出、前処理、後処理、全体を記録する。毎推論後の余分なrequestAnimationFrame待機を除き、表示だけ別のループで更新する。VRAM・GPU使用率はブラウザから測定済みと扱わない。
## 検証
`Verify Phase 1`は構文・座標・足の対応・欠損とFP16の数値処理を検査する。実ONNXをActionsでダウンロードし、ハッシュ、入力2つ、出力5つ、サンプル動画0/2/5秒の推論と全70点の有限値を確認する。実出力fixtureと独立計算した座標をブラウザ用デコーダーのテストへ渡す。
別ジョブはActionsのソフトウェアWebGPUで、主画面の初回推論と数値照合、動画入力、70点のJSON、任意の検出器を読み込まないこと、再読み込み後のモデルキャッシュ再利用を検査する。スクリーンショット・視覚確認は実行しない。このジョブはRTX4070 Ti SUPER/5060 Tiの速度測定ではない。実機60fps、3台合計180推論/秒、単眼距離精度は未検証。
結果はActionsの各ジョブと`instanthmr-numerical-verification` / `instanthmr-browser-numerical` artifactを参照する。`verification/instant/`には公開サンプルのCPU結果を収録。従来RTMWのfixtureは履歴・回帰テスト用として残す。
## キャッシュと通信
モデルは固定revisionの公開URLから直接fetchし、バイト数とSHA-256を検査する。Cache Storageからの読み込み時も照合する。保存容量が足りない場合は取得済みモデルで推論を続行し、永続保存できたと偽らない。動画・関節データのアップロードAPIはない。モデル・実行エンジン・Webフォントの取得には外部通信が発生する。Sites・Pages・127.0.0.1のキャッシュはオリジンごとに分かれる。
## 将来の全身トラッキング
現在は単一動画の70関節確認。1〜4視点のカメラトラッキング、LAN内WebRTC、時刻同期、複数視点統合、SteamVR送出は後続フェーズ。4層の責務と原文は`dist/architecture.md` / `docs/user-requirements.md`、元の計画書は`docs/project-plan.md`に保管する。計画書の古いモデル固定条件より、今回のInstantHMRへの明示的置換指示を適用する。
ソースはApache-2.0。InstantHMRのソースと重みのライセンスは異なる。重みの配布元はSAM Licenseを指定している。出典と既存サンプルの情報は`THIRD_PARTY_NOTICES.md`を参照。
