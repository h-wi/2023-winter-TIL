# 2023-winter-TIL

**아주대학교 사이버보안학과 201920686 김태휘**

2023 Winter 모각소 @ Ajou Univ 

# 1주차 활동 정리

1. 기본적인 Image Recognition Architecture인 VGG, AlexNet 논문 정리
2. [DyTox Github Code](https://github.com/arthurdouillard/dytox)을 기반으로 iCaRL Reproduce 코드 작성
3. Image Augmentation 기법 중 하나인 CutMix 논문 정리 및 PPT 파일 제작

### Reprocude iCaRL
* [iCaRL Reproducing based on DyTox Github code](https://github.com/h-wi/2023-winter-TIL/tree/main/week1/iCaRL-Reprod_0.52)

### 논문 정리
* VGG & AlexNet : [VGG & AlexNet](https://github.com/h-wi/2023-winter-TIL/tree/main/week1/VGG_정리.pdf)
* CutMix : [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://github.com/h-wi/2023-winter-TIL/tree/main/week1/CutMix_정리.pdf)
* CutMix PPT 파일 : [CutMix 개인 PPT 파일 제작](https://github.com/h-wi/2023-winter-TIL/tree/main/week1/CutMix_발표자료.pdf)

# 2주차 활동 정리

1. Self-Attetnion Module 구현하기
2. Compact Convolutional Transformer (CCT) 논문 정리
3. ResNet strikes back 논문 정리 및 PPT 파일 제작

### Implementation of Self-Attention

* [Re-implement Self-Attention Mechanism in transformer](https://github.com/h-wi/2023-winter-TIL/tree/main/week2/attention.py)

### 논문 정리
* Compact Convolutional Transformer(CCT) : [Compact Convolutional Transformer(CCT)](https://github.com/h-wi/2023-winter-TIL/tree/main/week2/CCT_정리.pdf)
* ResNet strikes back : [ResNet strikes back: An improved training procedure in timm](https://github.com/h-wi/2023-winter-TIL/tree/main/week2/ResNet-strikes-back_정리.pdf)
* ResNet strikes back PPT 파일 : [ResNet strikes back 개인 PPT파일 제작](https://github.com/h-wi/2023-winter-TIL/tree/main/week2/ResNetStrikesBack_발표.pdf)

# 3주차 활동 정리

1. Pytorch를 기반으로 Multi-GPU 사용할 수 있게 하는 Distributed Data Parallel 코드 구현
2. Self-Attention에 대한 개념 정리
3. Self-Attention의 응용 개념인 Window Self-Attention과 KNN Self-Attention 구현하기
4. Object Detection의 Fundamental 논문인 Faster R-CNN 정리를 위한 PPT 파일 제작


### Implementation of Window Self Attention based on CCT

* [Re-implement Self-Attention Mechanism based on Compact Convolutional Transformer](https://github.com/h-wi/2023-winter-TIL/blob/main/week3/implementation-wsa-ksa.py)

### Implementation of Distributed Data Parallel based on CutMix Baseline

* [Implement Distributed Data Parallel based on CutMix Baseline](https://github.com/h-wi/2023-winter-TIL/tree/main/week3/implementation-ddp)

### 개념 정리

* [What is Self-Attention and Query, Key, Value?](https://github.com/h-wi/2023-winter-TIL/blob/main/week3/self-attention.pdf)

### 논문 정리

* Faster R-CNN PPT 파일 : [Faster R-CNN 개인 PPT파일 제작](https://github.com/h-wi/2023-winter-TIL/blob/main/week3/Faster_R-CNN_%EB%B0%9C%ED%91%9C%EC%9E%90%EB%A3%8C.pdf)

# 4주차 활동 정리

1. MMDetection Framework를 통해 Face Mask Dataset format annotation converting하기
3. YOLOv8 모델 Face Mask Dataset 기반으로 학습해보기
4. 딥러닝 모델 예측 분석기법 중 하나인 CAM, Grad-CAM 정리 및 PPT 파일 제작

### Object Detection Exercise

* [Convet Facemask (.xml) Dataset format to COCO format](https://github.com/h-wi/2023-winter-TIL/blob/main/week4/convert.py)

* [Custom Object Detection Using MMDetection Framework](https://github.com/h-wi/2023-winter-TIL/blob/main/week4/mmdetection/my_custom_config.py)

* [Custom Object Detection Using YOLOv8 Framework](https://github.com/h-wi/2023-winter-TIL/blob/main/week4/yolo8s.ipynb)

### 개념 정리

* [CAM, Grad-CAM 비교 분석 및 정리](https://github.com/h-wi/2023-winter-TIL/blob/main/week4/Grad-CAM.pdf)

### 논문 정리

* Grad-CAM PPT 파일 : [Grad-CAM 개인 PPT파일 제작](https://github.com/h-wi/2023-winter-TIL/blob/main/week4/GradCAM_presentation.pdf)

# 5주차 활동 정리

1. Robust Fine-tuning of Zero-shot model 논문 정리 및 PPT 파일 제작
2. 딥러닝 분석 프레임 워크 직접 활용하기

### 논문 정리

* [Robust fine-tuning of Zero-shot models](https://github.com/h-wi/2023-winter-TIL/blob/main/week5/Robust-fine-tuning-of-zero-shot-models.pdf)