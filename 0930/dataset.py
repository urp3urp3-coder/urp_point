# dataset.py (최종 단순화 버전)

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import os

# 새로운 CSV 파일에 맞춘 컬럼명 설정
# 사용자님이 patient_id와 hgb_value를 bboxes.csv에 추가했다고 가정
HB_COL = 'hgb_value'
PATIENT_ID_COL = 'patient_id'
BBOX_CSV_NAME = 'bboxes.csv' # <<< 이제 이 파일 하나만 사용

def load_conjunctival_data(data_base_path: Path):
    """
    HB 값이 포함된 새로운 bboxes.csv 파일을 로드하여 데이터를 준비합니다.
    """
    
    # 1. 수정된 bboxes.csv 파일 로드
    try:
        merged_df = pd.read_csv(data_base_path / BBOX_CSV_NAME)
    except FileNotFoundError:
        print(f"\nERROR: '{BBOX_CSV_NAME}' 파일을 찾을 수 없습니다. 파일을 데이터 경로에 넣어주세요.")
        return [], [], [], [], [] 

    # 2. 필수 컬럼 확인
    required_cols = ['filename', 'x1', 'y1', 'x2', 'y2', HB_COL, PATIENT_ID_COL]
    if not all(col in merged_df.columns for col in required_cols):
        print(f"\nERROR: CSV 파일에 {required_cols} 중 누락된 컬럼이 있습니다. 컬럼 이름을 확인해 주세요.")
        print(f"현재 컬럼: {list(merged_df.columns)}")
        return [], [], [], [], []

    # 3. 로드 확인
    print(f"Loading data from {len(merged_df)} records...")
    print(f"--- 로드 확인: {len(merged_df)}개의 이미지-HB 쌍이 준비되었습니다. ---")

    # 4. 이미지 로드 및 마스크 생성
    images, masks, y_per_image, group_ids = [], [], [], []
    file_names = []
    
    for _, row in merged_df.iterrows():
        
        # 이미지 파일 경로는 filename 컬럼 사용
        original_filename = row['filename'].replace('\\', os.sep) 
        img_path = data_base_path / original_filename
        
        # 이미지 로드
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # 바운딩 박스에서 마스크 생성 (x1, y1, x2, y2 사용)
        try:
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), (255), -1) 
            
        except (ValueError, KeyError, TypeError):
            continue
        
        # 데이터 리스트에 추가
        images.append(img)
        masks.append(mask)
        y_per_image.append(row[HB_COL]) 
        
        # patient_id가 float이면 int로 변환하여 그룹 ID로 사용
        try:
            # patient_id를 정수형 그룹 ID로 변환하여 사용 (float으로 저장되어 있을 수 있으므로)
            group_ids.append(int(row[PATIENT_ID_COL]))
        except (ValueError, TypeError):
            group_ids.append(str(row[PATIENT_ID_COL])) # 실패 시 문자열 그대로 사용
            
        file_names.append(original_filename)

    print(f"Successfully loaded {len(images)} image-mask pairs.")
    return images, masks, y_per_image, group_ids, file_names