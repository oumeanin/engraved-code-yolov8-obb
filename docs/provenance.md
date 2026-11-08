# 원본 출처와 변경 범위

기준은 최종 제출에 가까운 `paper_final` 보존 묶음입니다. 이전 날짜별 실험, OO 고정 임계값 실험, Counter만 비교하던 평가 코드는 공개본의 기본 실행 경로에서 제외했습니다.

## 원본과 공개본

| 원본 | 공개본에서의 용도 |
| --- | --- |
| code/unified_config.py | settings.py의 공통 조건과 증강 설정 |
| code/u00_train_c.py ~ u04_train_f.py | train.py의 논문 a~d 선택 |
| code/01_oversample.py | 고정 배수와 이미지별 최대 배수 규칙 |
| code/final_table.py | research_core.py의 원문 함수 |
| code/u_eval_all.py | 기본 검출 설정과 평가 흐름의 참고 |
| code/dataset.yaml | configs/classes.json의 28클래스 순서 |
| splits/*.json | 파일 내용을 바꾸지 않고 복사 |
| splits/restore_split.py | 분할 복원 흐름, 입력 경로·실패 처리 정리 |

원본 경로는 개인 PC의 절대경로 대신 보존 묶음 기준으로 표시했습니다. 원본 파일 해시는 [source_manifest.json](source_manifest.json)에 있습니다.

`research_core.py`는 `final_table.py`에서 순수 함수를 추출한 모듈입니다. import만으로 추론이나 CSV 생성을 하지 않습니다. 소스와 공개본 함수의 AST를 대조해 알고리즘이 바뀌지 않았는지 검사합니다.

명령행 어댑터는 새로 정리했습니다. 입력·출력 경로를 명시하고, 기존 출력 경로를 거부하며, 학습·추론·저장 예측 평가를 따로 실행합니다. 분할 복원은 누락·해시 불일치를 오류로 처리합니다. 원본 연구 파일은 수정하지 않았습니다.

별도의 공개 라이선스는 이 정리 과정에서 임의로 지정하지 않았습니다.
