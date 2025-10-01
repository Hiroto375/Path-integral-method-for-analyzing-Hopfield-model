2025年Sセメスター理論演習の経路積分解析で作成したコードの一覧である。



Hopfield.cpp -> Hopfieldモデルのdirect simulationにおけるmの時間発展を(alpha,step,mean,std)の形式で出力

Path_Hopfield.cpp -> Hopfieldモデルの経路積分解析におけるmの時間発展を(alpha,step,mean,std)の形式で出力

Heatmap.cpp -> Hopfieldモデルの経路積分解析におけるmの最終値を(alpha,m0,m_final)の形式で出力

Critical.cpp -> Hopfieldモデルの経路積分解析における記憶容量の臨界点を二分探索して出力



noise_Hopfield.cpp -> ノイズつきHopfieldモデルのdirect simulationにおけるmの時間発展を(alpha,step,mean,std)の形式で出力

Path_noise_Hopfield.cpp -> ノイズつきHopfieldモデルの経路積分解析におけるmの時間発展を(alpha,step,mean,std)の形式で出力

noise_Heatmap.cpp -> ノイズつきHopfieldモデルの経路積分解析におけるmの最終値を(alpha,m0,m_final)の形式で出力。sigmaは固定している

noise_Critical.cpp -> ノイズつきHopfieldモデルの経路積分解析における記憶容量の臨界点を各sigmaに対し二分探索して出力
