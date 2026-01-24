# 데이터 누수(Data Leakage) 분석 보고서

## 발견된 문제점

### 1. **현재 상호작용이 윈도우에 포함되는 문제** (Critical)

**위치**: `precompute_sim_neighbors.py` 라인 272-290

**문제 설명**:
- `idx` 위치의 상호작용에 대한 이웃을 계산할 때, 윈도우에 **현재 상호작용(`idx`) 자체가 포함**되어 있습니다.
- 이는 자기 자신의 정보를 사용하여 유사한 이웃을 찾는 것으로, 데이터 누수입니다.

**코드 흐름**:
```python
# 라인 267-269: 윈도우 초기화 (인덱스 0 ~ window_size-1)
for u, i in zip(user_seq[:window_size], item_seq[:window_size]):
    window.append((int(u), int(i)))
    _add(int(u), int(i))

# 라인 272: window_size-1부터 시작
for idx in tqdm(range(window_size - 1, n_train), desc="precompute neighbors"):
    u = int(user_seq[idx])  # 현재 상호작용
    i = int(item_seq[idx])
    
    # 라인 278-282: 윈도우 상태를 사용하여 이웃 계산
    # 이 시점에서 윈도우는 [0, idx] 범위의 모든 상호작용을 포함
    # 따라서 현재 상호작용(idx)도 포함됨!
    uu = _select_pos_and_neg(...)
```

**영향**:
- CL weight가 높을수록 (예: 100) 이 문제가 더 심각해집니다.
- 모델이 현재 상호작용의 정보를 직접적으로 활용하여 학습하게 되어 비현실적으로 높은 성능을 보일 수 있습니다.
- 이는 200-300% 성능 향상의 주요 원인일 가능성이 높습니다.

### 2. **Global 윈도우 타입의 누적 효과**

**위치**: `precompute_sim_neighbors.py` 라인 295-305

**문제 설명**:
- `window_type='global'`일 때, 윈도우는 모든 과거 상호작용을 누적합니다.
- 이는 시간적 순서를 고려하지 않고 모든 정보를 사용하는 것으로, 미래 정보는 사용하지 않지만 과도한 정보를 사용할 수 있습니다.

**코드**:
```python
# 라인 295-305: 윈도우 업데이트
if args.window_type == 'local':
    # Sliding window: 가장 오래된 상호작용 제거
    u_old, i_old = window.popleft()
    _remove(u_old, i_old)

# Global의 경우: 아무것도 제거하지 않음
# Add new interaction (both local and global)
u_new = int(user_seq[idx + 1])
i_new = int(item_seq[idx + 1])
window.append((u_new, i_new))
_add(u_new, i_new)
```

### 3. **이웃 임베딩 사용 시점 확인 필요**

**위치**: `train.py` 라인 352-355, 442-445

**확인 사항**:
- 이웃 임베딩을 가져올 때 `.detach()`를 사용하고 있지만, 현재 상호작용의 임베딩이 포함되어 있는지 확인이 필요합니다.

**코드**:
```python
# 라인 352-355: U-U CL에서
pos_local_embs = user_embeddings[pos_local_ids_t.clamp(min=0), :].detach()
pos_global_embs = user_embeddings[pos_global_ids_t.clamp(min=0), :].detach()
```

## 수정 방안

### 수정 1: 현재 상호작용을 윈도우에서 제외

`precompute_sim_neighbors.py`의 이웃 계산 전에 현재 상호작용을 임시로 제거:

```python
# 라인 272 이후
for idx in tqdm(range(window_size - 1, n_train), desc="precompute neighbors"):
    u = int(user_seq[idx])
    i = int(item_seq[idx])
    
    # 현재 상호작용을 임시로 제거
    _remove(u, i)
    
    # 이웃 계산
    uu = _select_pos_and_neg(...)
    ii = _select_pos_and_neg(...)
    
    # 다시 추가 (다음 반복을 위해)
    _add(u, i)
    
    # 저장
    uu_pos_n[idx], uu_pos_s[idx], uu_neg_n[idx], uu_neg_s[idx] = uu
    ii_pos_n[idx], ii_pos_s[idx], ii_neg_n[idx], ii_neg_s[idx] = ii
```

### 수정 2: 윈도우 초기화 및 업데이트 로직 개선

현재 상호작용을 포함하지 않도록 윈도우 상태를 관리:

```python
# 초기화: window_size-1까지만 추가
for u, i in zip(user_seq[:window_size-1], item_seq[:window_size-1]):
    window.append((int(u), int(i)))
    _add(int(u), int(i))

# 처리: idx의 상호작용을 먼저 추가한 후 이웃 계산
for idx in tqdm(range(window_size - 1, n_train), desc="precompute neighbors"):
    u = int(user_seq[idx])
    i = int(item_seq[idx])
    
    # 현재 상호작용을 윈도우에 추가 (이웃 계산 전)
    # 하지만 이웃 계산 시에는 제외해야 함
    # ... (수정 필요)
```

## 권장 사항

1. **즉시 수정**: 현재 상호작용을 이웃 계산에서 제외
2. **재실험**: 수정 후 CL weight=100으로 재실험하여 성능 변화 확인
3. **검증**: 수정 전후의 이웃 리스트를 비교하여 변경사항 확인

## 추가 확인 사항

### 이웃 임베딩 사용 확인 (정상)

**위치**: `train.py` 라인 395, 404, 485, 494

**확인 결과**:
- 이웃 임베딩을 가져올 때 `.detach()`를 사용하여 그래디언트 전파를 차단하고 있습니다.
- 이웃 임베딩은 `user_embeddings`와 `item_embeddings`의 **업데이트 전 상태**에서 가져옵니다 (라인 503-504에서 업데이트됨).
- 따라서 이 부분은 데이터 누수 문제가 없습니다.

**코드 순서**:
1. 라인 316-317: 현재 상호작용에 대한 임베딩 업데이트 계산
2. 라인 319-501: CL loss 계산 (이웃 임베딩은 업데이트 전 상태 사용)
3. 라인 503-504: 임베딩 행렬 업데이트

## 적용된 수정

### 수정 내용
`precompute_sim_neighbors.py`에서 이웃 계산 전에 현재 상호작용을 카운터에서 임시로 제거하도록 수정했습니다.

**수정 전**:
- 현재 상호작용(`idx`)이 윈도우에 포함된 상태에서 이웃 계산
- 자기 자신의 정보를 사용하여 유사한 이웃을 찾음 (데이터 누수)

**수정 후**:
- 이웃 계산 전에 현재 상호작용을 카운터에서 제거
- 이웃 계산 후 다시 추가 (다음 반복을 위해)

## 예상 결과

수정 후에는:
- CL weight=100에서의 비현실적인 성능 향상(200-300%)이 사라질 것으로 예상됩니다.
- 정상적인 범위의 성능 향상 (예: 5-20%)만 남을 것으로 예상됩니다.
- 모델의 실제 일반화 성능을 정확히 평가할 수 있게 됩니다.

## 다음 단계

1. **precomputed neighbors 재생성**: 수정된 코드로 `precompute_sim_neighbors.py`를 실행하여 새로운 이웃 파일 생성
2. **모델 재학습**: 수정된 이웃 파일을 사용하여 모델 재학습
3. **성능 비교**: 수정 전후의 성능 비교 (특히 CL weight=100일 때)
4. **검증**: 이웃 리스트가 실제로 변경되었는지 확인
