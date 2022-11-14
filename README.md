# 국방 AI 경진대회 코드 사용법
- MMC_Lab팀, 박성욱, 문희찬, 이승리, 임수빈
- 닉네임 : 욱스박스,무니무니찬,victory,teol
- 최종 순위 : 9등

# 핵심 파일 설명
  - 학습 데이터 경로: `./data`
  - Network 초기 값으로 사용한 공개된 Pretrained 파라미터: `architecture: DeepLabV3Plus`
`encoder: resnext50_32x4d`
`encoder_weight: imagenet` (대회에서 제공한 baseline 코드에서 `train.yaml` 파일의 세팅사항)
  - 공개 되어있는 Pretrained 모델 기반으로 추가 Fine Tuning 학습을 한 파라미터 3개
    - `./results/train/20221110_072024/model.pt`
    - `./results/train/20221114_101319/model.pt`
    - `./results/train/20221114_142823/model.pt`
  - 학습 실행 스크립트: `python train.py`
  - 학습 메인 코드: `./train.py`
  - 테스트 실행 스크립트: `python predict.py`
  - 테스트 메인 코드: `./predict.py`
  - 테스트 이미지, 마스크 경로: `./data/test`
  - 테스트 결과 이미지 경로: `./results/pred/20221110_072024/final_pred/mask`

## 코드 구조 설명
- segmentation-models-pytorch library 사용(ensemble한 모든 모델 동일)
    - 최종 사용 모델 :
        - architecture: DeepLabV3Plus
        - encoder: resnext50_32x4d
        - encoder_weight: imagenet
    - data augmentation 추가(train.py에 추가)
    ```
    ./results/train/20221110_072024/model.pt 모델 학습 시 
  
    aug = A.Compose([
        A.augmentations.transforms.PixelDropout(p=0.3),
        A.VerticalFlip(p=0.5),
    ])
    val_aug = A.Compose([
        A.VerticalFlip(p=0.5,always_apply=False)
    ])
    ```
    ```
    ./results/train/20221114_101319/model.pt 모델 학습 시 
  
    aug = A.Compose([
        A.augmentations.transforms.Downscale(scale_min=0.15, scale_max=0.15,p=0.3),
        A.augmentations.transforms.PixelDropout(p=0.3),
        A.VerticalFlip(p=0.5),
    ])
    val_aug = A.Compose([
        A.VerticalFlip(p=0.5,always_apply=False)
    ])
    ```
    ```
    ./results/train/20221114_142823/model.pt 모델 학습 시 
  
    aug = A.Compose([
        A.RandomContrast(p=0.3),
        A.augmentations.transforms.Downscale(scale_min=0.15, scale_max=0.15,p=0.3),
        A.augmentations.transforms.PixelDropout(p=0.3),
        A.VerticalFlip(p=0.5),
    ])
    val_aug = A.Compose([
        A.VerticalFlip(p=0.5,always_apply=False)
    ])
    ```
  
- **최종 제출 파일 : final_pred_mask.zip**
- **학습된 가중치 파일 : 핵심파일 설명부분에 기재된 세개의 파라미터와 동일**

## 주요 설치 library
- torchmetrics==0.10.2
- torch==1.13.0
- torchvision == 0.14.0
- mmcv-full==1.6.0
- segmentation-models-pytorch==0.3.0
- opencv-python==4.6.0.66
- numpy==1.23.4

# 실행 환경 설정

  - 소스 코드 및 conda 환경 설치
    ```
    unzip mmc_lab.zip -d mmc_lab_code
    cd ./mmc_lab_code

    conda env create -n aiconnect
    conda activate aiconnect
    pip install -r requirements.txt
    ```
    
# 학습 실행 방법

  - 학습 데이터 경로 설정
    - 위의 실행 환경 설정을 하면 따로 학습 데이터 경로 설정 필요 없음

  - 학습 스크립트 실행
    ```
    python train.py
    ```


# 테스트 실행 방법

  - 테스트 스크립트 실행
    ```
    python predict.py
    
    # 상기의 3가지 추론 결과를 Pixel-wise Averaging 처리하여 최종 detection 결과 생성
    python predict_en.py
    ```
