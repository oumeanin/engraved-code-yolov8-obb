# 실행 가이드

모든 명령은 저장소 루트에서 실행합니다. 데이터·라벨·가중치는 별도로 준비해야 하며, 아래 명령을 실제 연구 데이터로 재실행한 결과를 제공하는 것은 아닙니다.

## 1. 실행 환경 준비

가상환경을 만들고 사용 중인 장비에 맞는 PyTorch/CUDA 환경을 준비합니다.

```bash
python -m venv ".venv"
```

Windows PowerShell에서는 다음과 같이 활성화합니다.

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS에서는 다음을 사용합니다.

```bash
source ".venv/bin/activate"
```

```bash
python -m pip install -r "requirements.txt"
```

`requirements.txt`는 직접 의존성 목록이며 전체 환경의 lockfile이 아닙니다. 보존 보고서의 버전과 코드 검사 환경은 [환경 기록](environment.md)에 있습니다.

## 2. 데이터 형식과 클래스 순서

원본 데이터는 이미지 500장과 28개 문자 클래스의 회전 박스 라벨입니다. 클래스 ID는 [classes.json](../configs/classes.json)의 순서를 유지합니다.

정답 라벨 한 줄의 형식은 다음과 같습니다. 좌표는 정규화된 네 꼭짓점이며, 예측 TXT에는 마지막에 신뢰도가 추가됩니다.

```text
class_id x1 y1 x2 y2 x3 y3 x4 y4
class_id x1 y1 x2 y2 x3 y3 x4 y4 confidence
```

```text
data/origin/
  images/train/  images/val/  images/test/
  labels/train/  labels/val/  labels/test/
```

Train/Validation/Test는 분할별 350/75/75입니다. 네 분할은 같은 500장의 원본 분할과 무작위 재분할 세 종류이며, 교차검증이나 서로 독립적인 네 데이터셋을 의미하지 않습니다.

## 3. 보존 명세로 분할 복원

`--source-pool`에는 원본 이미지와 라벨이 `images/<split>/`, `labels/<split>/` 구조로 들어 있는 데이터셋 경로를 지정합니다.

```bash
python "code/restore_split.py" --manifest "splits/origin.json" --source-pool "/path/to/source-pool" --output "data/origin"
```

재분할은 명세를 `resplit_A.json`, `resplit_B.json`, `resplit_C.json`으로 바꾸고 서로 다른 출력 경로를 사용합니다. 명세의 해시는 파일명 목록을 검사하며 이미지·라벨 내용의 해시가 아닙니다.

## 4. 오버샘플링 조건 준비

논문의 (b), (d)를 실행할 때는 먼저 새 데이터 사본을 만듭니다.

```bash
python "code/oversample.py" --source "data/origin" --output "data/origin_oversampled"
```

| 원본 포함 총 배수 | 클래스 |
| --- | --- |
| 10 | ZZ, TT, SS, 11, VV |
| 7 | YY, 55 |
| 5 | NN, DD, GG, MM, WW, PP, FF |

한 이미지에 여러 대상 클래스가 있으면 지정 배수의 최댓값을 사용합니다. 이미지와 라벨 전체를 함께 복제하며 복제 단계에서 이미지 변형은 하지 않습니다. 같은 이미지에 포함된 다른 문자도 함께 늘어납니다. Validation/Test는 추가 복제하지 않습니다.

## 5. 학습 조건 선택

[dataset.example.yaml](../configs/dataset.example.yaml)의 `path`를 실제 데이터셋으로 지정합니다. 상대 `path`는 YAML이 있는 폴더를 기준으로 해석합니다. 기존 28클래스 순서는 바꾸지 않습니다.

| `--experiment` | 논문 조건 | 입력 데이터 | 증강 |
| --- | --- | --- | --- |
| a | 기준 | 원본 | OFF |
| b | 오버샘플링 | 복제 사본 | OFF |
| c | 증강 | 원본 | ON |
| d | 결합 | 복제 사본 | ON |

```bash
python "code/train.py" --experiment c --data "configs/dataset.example.yaml" --weights "/path/to/yolov8n-obb.pt" --output "runs/c_origin"
```

오버샘플링 조건에서는 예시 YAML을 별도 파일로 복사하고 `path`를 `../data/origin_oversampled`로 바꾼 뒤, 해당 YAML을 `--data`에 지정합니다. `--experiment b` 또는 `d`를 선택하는 것만으로 데이터 복제가 자동 실행되지는 않습니다.

DOTA 사전학습 YOLOv8n-OBB 가중치를 로컬에 준비합니다. 없는 가중치를 자동으로 다운로드하지 않습니다.

공통 설정은 epochs 100, batch 4, imgsz 1280, patience 15, seed 0입니다. 증강 ON은 회전 ±10°, HSV 채도·명도 0.7/0.4, 이동 0.05, 크기 변화 0.1을 사용합니다. 반전·모자이크·믹스업은 사용하지 않습니다.

## 6. 기본 예측 저장

Test 예측을 생성합니다.

```bash
python "code/predict.py" --weights "runs/c_origin/weights/best.pt" --images "data/origin/images/test" --output "predictions/c_origin_test"
```

신뢰도 임계값 비교에는 같은 분할의 Validation 예측도 준비합니다.

```bash
python "code/predict.py" --weights "runs/c_origin/weights/best.pt" --images "data/origin/images/val" --output "predictions/c_origin_val"
```

기본 검출은 imgsz 1280, conf 0.30, 클래스별 NMS IoU 0.20을 사용합니다. 출력의 `labels` 폴더에 신뢰도를 포함한 예측 TXT가 저장됩니다.

## 7. 저장 예측 평가

평가 명령은 저장된 TXT를 읽으며 모델 추론을 실행하지 않습니다.

| `--post` | 출력에 포함되는 결과 | Validation 입력 |
| --- | --- | --- |
| none | 기본 결과 | 불필요 |
| confidence | 기본 결과와 클래스별 신뢰도 임계값 | 필요 |
| nms | 기본 결과와 클래스 무관 NMS | 불필요 |
| all | 기본 결과와 두 후처리 각각의 결과 | 필요 |

```bash
python "code/evaluate.py" --gt "data/origin/labels/test" --pred "predictions/c_origin_test/labels" --post nms --output "results/c_origin_nms.json"
```

```bash
python "code/evaluate.py" --gt "data/origin/labels/test" --pred "predictions/c_origin_test/labels" --val-gt "data/origin/labels/val" --val-pred "predictions/c_origin_val/labels" --post all --output "results/c_origin_all.json"
```

후처리 (e)는 Validation에서 임계값을 결정하고 고정해 Test에 적용합니다. 후처리 (f)는 신뢰도 내림차순으로 예측을 남기며, 이미 남긴 박스와 polygon IoU 0.5 이상 겹친 예측을 클래스와 무관하게 제거합니다. `all`에서도 두 후처리를 연속 적용하지 않습니다.

출력 JSON의 `exact_images`는 완전일치 이미지 수, `images`는 평가 이미지 수, `pct`는 퍼센트입니다. `clopper_pearson_95_ci`는 0~1 비율입니다. 이 명령은 한 분할을 평가하며 논문의 네 분할 평균±표준편차·통합 신뢰구간을 자동 집계하지 않습니다.

완전일치 매칭 순서와 과거 코드의 조건명은 [논문과 코드의 대응](method_mapping.md)에 있습니다.

## 8. 코드 검사

학습·추론 없이 합성 입력 단위 검사만 실행하려면 다음을 사용합니다.

```bash
python -m pip install -r "requirements-test.txt"
python -m unittest discover -s "tests" -v
```

기존 출력 파일·폴더는 덮어쓰지 않으므로 명령을 다시 실행할 때는 새 출력 경로를 지정합니다. CLI 도움말은 다음처럼 확인합니다.

```bash
python "code/evaluate.py" --help
```

검사의 의미와 미확인 항목은 [검증 및 재현 범위](reproduction_status.md)에 있습니다.
