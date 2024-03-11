# FSD-Dataset
 This repository presents a subset of the FSD dataset for song deepfake detection.
 FSD is our our work titled "FSD: An Initial Chinese Dataset for Fake Song Detection," which was available on arxiv at "https://arxiv.org/abs/2309.02232".
## Update
We have released the best song-trained ADD model, W2V2-LCNN, as outlined in the paper. The output logits can be seen in `/Inference_score`.
![](./img/table4.png). The speech-trained ADD model can be seen in this repository [ADD-W2V2-LCNN-19LA0.6](https://github.com/xieyuankun/ADD-W2V2-LCNN-19LA0.6)
## Inference
Run `python generate_FSD_online.py` to generate the result txt. 

For EER result, run `python evaluate_FSD.py`.

Test the model on your dataset, please modify `/wav2vec2_xls-r300-song/raw_dataset.py`

Line28 `self.path_to_audio = '/data2/xyk/evalvocal/F01/wav'`

Line29 `self.path_to_protocol = '/data2/xyk/evalvocal/F01/label.txt'`
 



## Reference
- [The implementation of F01-F03 methods](https://github.com/svc-develop-team/so-vits-svc)
- [The implementation of F04 method](https://github.com/MoonInTheRiver/DiffSinger)
- [The implementation of F05 method](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
- [The implementation of audio source separation tool](https://github.com/Anjok07/ultimatevocalremovergui)
- [The code structrue of audio deepfake detection model](https://github.com/yzyouzhang/ASVspoof2021_AIR)