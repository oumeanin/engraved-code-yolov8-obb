# 음각 마킹 코드 인식

한국정보처리학회 ACK 2026 논문 **「클래스 무관 비최대억제를 적용한 YOLOv8-OBB 기반 음각 마킹 코드 인식」**의 최종 실험 코드를 정리한 저장소입니다.

- 기준 자료: 최종 제출에 가까운 `paper_final` 코드·설정·분할 명세

금속 부품의 음각 문자에 대해 증강과 이미지 단위 오버샘플링의 효과를 비교하고, 서로 다른 클래스의 중복 검출을 클래스 무관 NMS로 억제합니다. 원본 연구 함수를 보존하면서 입력·출력 경로와 실행 명령을 정리했습니다.

데이터 이미지·라벨, 사전학습/학습 가중치, 예측 파일은 포함하지 않습니다. 실제 학습과 논문 전체 결과를 이 정리본으로 다시 실행한 것은 아닙니다. [검증 및 재현 범위](docs/reproduction_status.md)를 먼저 확인하세요.

## 구성

```text
code/
  settings.py          공통 학습 조건·증강·실험 조건
  research_core.py     원문에서 추출한 IoU·평가·후처리 함수
  train.py             한 학습 조건 실행
  predict.py           기본 검출 결과 저장
  evaluate.py          저장된 예측의 평가와 후처리
  oversample.py        새 데이터 사본에 이미지 단위 차등 복제
  restore_split.py     보존 명세로 데이터 분할 복원
configs/
  classes.json         기존 28클래스 순서
  dataset.example.yaml 데이터 경로 예시
splits/                기존 네 분할 JSON과 요약
tests/                 합성 입력을 이용한 코드 검사
docs/                  논문 대응·출처·검증 범위
```

## 논문 조건과 코드

| 논문 | 조건 | 과거 코드 이름 |
| --- | --- | --- |
| (a) | 기준, 증강 OFF | c / HighRes-1280 |
| (b) | 오버샘플링, 증강 OFF | d / oversample |
| (c) | 증강 | e / augmentation |
| (d) | 오버샘플링 + 증강 | f / oversample_aug |
| (e) | Validation에서 정한 클래스별 신뢰도 임계값 | g |
| (f) | 클래스 무관 NMS | g2 또는 g′ |

학습 조건 (a)~(d)는 모두 입력 크기 1280입니다. 후처리 (e)와 (f)는 기본 예측에 **별도로** 적용합니다. 논문에서 (e)는 기반 모델 (c), (d)에 비교했습니다.

## 설치

아래 명령은 저장소 루트에서 실행합니다. 이 정리본의 단위 검사는 Python 3.12에서 확인합니다. 모델 학습용 PyTorch/CUDA 환경은 사용하는 장비에 맞게 별도로 준비해야 합니다.

```bash
python -m venv ".venv"
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source ".venv/bin/activate"
python -m pip install -r "requirements.txt"
```

`requirements.txt`는 직접 의존성 목록이며 전체 환경의 lockfile이 아닙니다. 보존 보고서에 기록된 버전과 이번 단위 검사 환경은 [환경 기록](docs/environment.md)에서 구분합니다.

학습·추론을 제외한 단위 검사만 설치하려면 다음을 사용합니다.

```bash
python -m pip install -r "requirements-test.txt"
python -m unittest discover -s "tests" -v
```

## 데이터 구성과 분할

기존 클래스 ID와 `configs/classes.json` 순서를 유지한 YOLO OBB 라벨이 필요합니다. 정답 라벨 한 줄은 클래스 ID와 정규화된 네 꼭짓점 좌표입니다. 저장 예측은 같은 형식에 신뢰도 값이 추가됩니다.

```text
data/origin/
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

원본 이미지·라벨 풀을 준비한 뒤 보존된 명세로 분할을 복원할 수 있습니다. 아래 원본 풀 경로는 실제 파일이 있는 위치로 바꿉니다.

```bash
python "code/restore_split.py" --manifest "splits/origin.json" --source-pool "/path/to/source-pool" --output "data/origin"
```

`resplit_A.json`, `resplit_B.json`, `resplit_C.json`도 같은 방식으로 사용할 수 있습니다. 분할당 Train/Validation/Test는 350/75/75이며, 같은 500장을 네 가지로 나눈 명세입니다. 명세의 해시는 파일명 목록을 확인하며 이미지·라벨 내용 해시를 뜻하지 않습니다.

## 차등 오버샘플링

```bash
python "code/oversample.py" --source "data/origin" --output "data/origin_oversampled"
```

클래스별 고정 복제 배수는 다음과 같습니다.

| 배수 | 클래스 |
| --- | --- |
| 10 | ZZ, TT, SS, 11, VV |
| 7 | YY, 55 |
| 5 | NN, DD, GG, MM, WW, PP, FF |

한 이미지에 대상 클래스가 여러 개이면 가장 큰 배수를 적용합니다. 원본을 포함한 총 배수이며 이미지와 라벨 전체를 함께 복제합니다. 복제 단계에는 회전·밝기 변형이 없습니다. 같은 이미지의 다른 문자도 함께 복제됩니다. Validation/Test는 추가 복제하지 않습니다.

## 학습

`configs/dataset.example.yaml`의 데이터 경로를 실제 데이터로 지정합니다. 상대 `path`는 YAML이 있는 폴더를 기준으로 해석합니다. 오버샘플링 조건은 복제 데이터셋을 가리키는 별도 YAML을 사용합니다. 예시의 클래스 순서는 바꾸지 않습니다.

```bash
python "code/train.py" --experiment c --data "configs/dataset.example.yaml" --weights "/path/to/yolov8n-obb.pt" --output "runs/c_origin"
```

`--experiment`는 논문의 `a`, `b`, `c`, `d`입니다. DOTA 사전학습 YOLOv8n-OBB 가중치를 로컬에 준비해야 합니다. 명령은 없는 가중치를 자동으로 다운로드하지 않습니다.

주요 학습 설정은 epochs 100, batch 4, imgsz 1280, patience 15, seed 0입니다. 증강은 회전 ±10°, HSV 채도·명도 0.7/0.4, 이동 0.05, 크기 변화 0.1이며 반전·모자이크·믹스업을 사용하지 않습니다.

## 기본 예측

```bash
python "code/predict.py" --weights "runs/c_origin/weights/best.pt" --images "data/origin/images/test" --output "predictions/c_origin_test"
```

입력 크기 1280, conf 0.30, 기본 클래스별 NMS IoU 0.20을 사용하고 신뢰도를 포함한 예측 TXT를 저장합니다.

## 저장 예측 평가와 후처리

모델을 실행하지 않고 이미 저장된 예측 TXT만 읽습니다. 클래스 무관 NMS의 추가 IoU 임계값은 0.5입니다.

```bash
python "code/evaluate.py" --gt "data/origin/labels/test" --pred "predictions/c_origin_test/labels" --post nms --output "results/c_origin_nms.json"
```

기본 결과는 `--post none`을 사용합니다. 클래스별 신뢰도 임계값에는 같은 분할의 Validation 정답과 예측을 함께 지정합니다.

```bash
python "code/evaluate.py" --gt "data/origin/labels/test" --pred "predictions/c_origin_test/labels" --val-gt "data/origin/labels/val" --val-pred "predictions/c_origin_val/labels" --post all --output "results/c_origin_all.json"
```

`all`은 기본 결과, 신뢰도 임계값, 클래스 무관 NMS를 각각 평가합니다. 신뢰도 임계값을 적용한 뒤 NMS를 연속 적용하는 명령이 아닙니다. Validation 예측은 `predict.py`의 입력을 Validation 이미지 폴더로 바꾸어 별도로 준비합니다.

출력 JSON은 지정한 한 분할의 결과입니다. `pct`는 퍼센트이고 `clopper_pearson_95_ci`는 0~1 비율입니다. 논문의 네 분할 평균±표준편차·통합 신뢰구간 집계를 이 명령의 단일 출력과 혼동하지 않습니다.

완전일치는 클래스와 OBB polygon IoU 0.5를 기준으로 정답·예측을 일대일로 매칭하고, 정답 또는 예측이 남으면 실패로 판정합니다. 공개본은 기존의 **최대 IoU 선택 후 클래스 확인** 순서를 보존합니다. 자세한 정책은 [구현 대응](docs/method_mapping.md)에 있습니다.

기존 출력 경로는 재사용하지 않습니다. 새 출력 경로를 지정하세요. CLI 도움말은 `python "code/evaluate.py" --help`처럼 확인할 수 있습니다.

## 연구 결과와 재현 상태

논문에 보고된 증강 단독 (c)의 Test 완전일치율은 **95.7%**입니다. 이 값은 논문에서 인용한 값이며 이번 코드 정리 중 새로 측정한 값이 아닙니다. 논문 표의 표기값은 [paper_tables.json](docs/paper_tables.json)에 보존했습니다.

보존 집계 코드에는 논문 (c)/(d)의 후처리 결과가 연결돼 있지만, (a)/(b)를 포함한 네 모델·16회 NMS 적용 전체를 만드는 실행 자료는 조사한 묶음에서 확인되지 않았습니다. 이 정리본의 평가 명령이 모든 조건의 저장 예측을 처리할 수 있다는 사실과 논문 전체 결과가 재현됐다는 주장을 구분합니다.

원본 출처와 변경 범위는 [provenance.md](docs/provenance.md), 실제 검증 범위는 [reproduction_status.md](docs/reproduction_status.md)에 기록합니다.
