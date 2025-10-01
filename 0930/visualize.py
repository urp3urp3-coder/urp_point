# visualize.py

import numpy as np, cv2
from sklearn.cluster import KMeans
from pathlib import Path
import os
# 전처리 함수를 가져오기 위해 point.py에서 필요한 import
from point import preprocess, rng 
from dataset import load_conjunctival_data # <<< 데이터 로딩 함수 import

# --- 1. 글레어 제거 함수 ---
def remove_glare(mask, img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    glare = ((hsv[...,1] < 20) & (hsv[...,2] > 240)).astype(np.uint8) 
    cleaned = mask.copy().astype(np.uint8)
    cleaned[glare>0] = 0
    return cleaned


# --- 2. K-means 샘플링에 필요한 유틸리티 함수 ---
def features_xy_color(img_bgr, ys, xs, alpha_xy=0.5):
    H, W = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    coords = np.stack([xs/W, ys/H], 1).astype(np.float32) 
    ab = lab[ys, xs][:,1:3]
    hsvp = hsv[ys, xs]
    feats = np.concatenate([alpha_xy*coords, ab, hsvp], 1)
    return feats


def kmeans_sampling(img_bgr, mask, k=5, alpha_xy=0.5, extra_perc=0.0):
    m = mask.astype(bool)
    ys, xs = np.where(m)
    
    if len(xs) == 0:
        return np.empty((0,7), np.float32), np.empty((0,2), np.int32)
        
    feats = features_xy_color(img_bgr, ys, xs, alpha_xy=alpha_xy)

    k_actual = int(min(max(2, k), len(feats)))
    km = KMeans(n_clusters=k_actual, n_init="auto", random_state=0) 
    labels = km.fit_predict(feats)
    centers = km.cluster_centers_

    picked = []
    for c in range(k_actual):
        idxs = np.where(labels==c)[0]
        if len(idxs)==0: continue
        dif = feats[idxs] - centers[c]
        j = idxs[np.argmin((dif*dif).sum(1))] 
        picked.append(j)
        
        if extra_perc > 0 and len(idxs) > 1:
            n_extra = max(1, int(len(idxs)*extra_perc))
            extra = rng.choice(idxs, size=min(n_extra, len(idxs)), replace=False)
            picked.extend(extra.tolist())

    picked = np.unique(np.array(picked))
    sel_feats = feats[picked]
    sel_xy = np.stack([ys[picked], xs[picked]], 1)
    
    return sel_feats.astype(np.float32), sel_xy.astype(np.int32)


# --- 3. 메인 시각화 함수 (save_sub_folder 인자 추가) ---
def visualize_sampling(img_bgr, mask, file_id, save_sub_folder, config_num=5):
    """
    하나의 이미지에 대해 특정 Config의 K-means 샘플링을 시각화하고 저장합니다.
    """
    
    if config_num == 2:
        k=5; extra_perc=0.0
        config_name = "2_KMeans_Preproc"
    elif config_num == 3:
        k=5; extra_perc=0.0 
        config_name = "3_KMeans_DM"
    elif config_num == 4:
        k=5; extra_perc=0.2
        config_name = "4_KMeans_DM_Extra"
    elif config_num == 5:
        k=5; extra_perc=0.2
        config_name = "5_KMeans_DM_Extra_Jitter"
    else:
        print(f"Config {config_num} does not use point sampling (or is invalid). Visualization aborted.")
        return

    print(f"\n[ID: {file_id}] --- Starting Visualization for {config_name} ---")
    
    x = preprocess(img_bgr) 
    m2 = remove_glare(mask, x) 
    
    if config_num == 2:
        sampling_mask = mask
    else:
        sampling_mask = m2 

    sel_feats, sel_xy = kmeans_sampling(x, sampling_mask, k=k, alpha_xy=0.5, extra_perc=extra_perc)
    
    print(f"Total points sampled: {len(sel_xy)} (for {config_name})")
    
    if len(sel_xy) == 0:
        print("No valid ROI or sampled points found in the image.")
        return

    vis_img = x.copy()
    
    if config_num in [4, 5]:
        dot_radius = 1 
    else:
        dot_radius = 10 
    
    dot_color = (255, 0, 0) # 파란색 (BGR 순서)

    for y, x_coord in sel_xy:
        cv2.circle(vis_img, (x_coord, y), dot_radius, dot_color, -1) 
        
    base_dir_name = "visualization_results"

    # 수정된 부분: save_sub_folder를 사용하여 Country/PatientID 구조 생성
    output_dir_str = os.path.join(os.getcwd(), base_dir_name, "KMeans_Sampling_Results", save_sub_folder)
    output_filename = f"{config_name}_{file_id}.bmp"

    Path(output_dir_str).mkdir(parents=True, exist_ok=True) 
    save_path = os.path.join(output_dir_str, output_filename)
    vis_img_uint8 = vis_img.astype(np.uint8)

    success = cv2.imwrite(save_path, vis_img_uint8)
    
    if success:
        # 출력 경로도 새로운 구조에 맞게 수정
        print(f"Visualization saved to '{base_dir_name}/KMeans_Sampling_Results/{save_sub_folder}/{output_filename}'.")
    else:
        print(f"!!! FATAL ERROR: cv2.imwrite failed for file ID: {file_id}. Check permissions or disk space.")

        
# ---------------------------------------------
# 4. Main 실행 블록 (중복 제거 로직 및 폴더 구조 생성)
# ---------------------------------------------

def main_run_visuals():
    data_base_path = Path('../Conjunctival Images for Anemia Detection/') 
    
    print("--- Starting Data Loading for Visualization ---")
    images, masks, y_per_image, group_ids, file_names = load_conjunctival_data(data_base_path)

    if not images:
        print("\nERROR: No data was loaded successfully. Cannot visualize.")
        return
        
    total_samples = len(images)
    print(f"Data loading complete. Total {total_samples} raw samples found.")

    # -----------------------------------------------------------------
    # ✨ 파일 이름 기반 중복 제거 및 메타데이터 추출 (Country/PatientID/FileName)
    # -----------------------------------------------------------------
    unique_data = {}
    
    for i, file_name in enumerate(file_names):
        # 1. 파일 이름에서 Country 추출 (예: 'India\11\...')
        parts = file_name.replace('\\', '/').split('/')
        if len(parts) < 2: continue
            
        country = parts[0]
        # group_ids (patient_id)는 dataset.py에서 로드됨
        patient_id = group_ids[i] 
        
        # 2. 고유 키 생성 (같은 이미지가 여러 BBOX 레코드를 가질 때, BBOX 레코드 ID까지 필요하므로 file_name으로 고유 키를 만듦)
        unique_key = f"{country}/{patient_id}/{file_name}" 
        
        if unique_key not in unique_data:
            unique_data[unique_key] = {
                'image': images[i], 
                'mask': masks[i], 
                # 저장에 사용할 폴더명/파일명 정보
                'country': country,
                'patient_id': patient_id,
                # 순수 파일명(확장자X)을 저장 이름의 일부로 사용
                'save_name': os.path.splitext(file_name.split(os.sep)[-1])[0] 
            }
            
    # 시각화에 필요한 고유한 데이터 리스트 재구성
    images_unique = [d['image'] for d in unique_data.values()]
    masks_unique = [d['mask'] for d in unique_data.values()]
    meta_data_unique = [{k:d[k] for k in ['country', 'patient_id', 'save_name']} for d in unique_data.values()]
    
    num_unique = len(images_unique)
    print(f"Filtered: {num_unique} unique image records will be processed for visualization.")
    print("\n--- Starting Visualization for UNIQUE IMAGES ---")
    
    # 고유한 이미지 레코드에 대해 반복
    for i, (img_bgr, mask, meta) in enumerate(zip(images_unique, masks_unique, meta_data_unique)):
        
        # 저장 파일명에 사용할 고유 ID (Country_PatientID_FileName)
        save_file_id = f"{meta['country']}_{meta['patient_id']}_{meta['save_name']}" 
        
        # 저장할 하위 폴더 경로: Country/PatientID
        save_sub_folder = f"{meta['country']}/{meta['patient_id']}"
        
        print(f"\n=======================================================")
        print(f"Processing Image {i+1}/{num_unique} (Patient: {meta['country']}/{meta['patient_id']})")
        print(f"=======================================================")

        # save_sub_folder를 visualize_sampling에 전달
        # 이제 폴더가 겹치지 않아 이미지 덮어쓰기 문제 해결
        visualize_sampling(img_bgr, mask, save_file_id, save_sub_folder, config_num=2) 
        visualize_sampling(img_bgr, mask, save_file_id, save_sub_folder, config_num=3) 
        visualize_sampling(img_bgr, mask, save_file_id, save_sub_folder, config_num=4) 
        visualize_sampling(img_bgr, mask, save_file_id, save_sub_folder, config_num=5) 
    
    print("\nVisualization process finished.")

if __name__ == '__main__':
    main_run_visuals()