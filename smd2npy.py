import os
import numpy as np
import pandas as pd

# 경로 설정
base_path = 'C:/Users/hyung/Anomaly-Transformer/dataset/OmniAnomaly/ServerMachineDataset'
output_path = 'C:/Users/hyung/Anomaly-Transformer/dataset/SMD'
os.makedirs(output_path, exist_ok=True)

def load_and_concat(folder_name):
    folder = os.path.join(base_path, folder_name)
    all_data = []
    for filename in sorted(os.listdir(folder)):
        if filename.startswith('machine-1-') and filename.endswith('.txt'):
            file_path = os.path.join(folder, filename)
            df = pd.read_csv(file_path, header=None, sep=None, engine='python')  # 자동 구분자 감지
            all_data.append(df)
    return pd.concat(all_data, axis=0, ignore_index=True)

# ✅ 데이터 로딩
train_df = load_and_concat('train')
test_df = load_and_concat('test')
label_df = load_and_concat('test_label')

# ✅ CSV 저장
train_df.to_csv(os.path.join(output_path, 'SMD_train_l.csv'), index=False, header=False)
test_df.to_csv(os.path.join(output_path, 'SMD_test_l.csv'), index=False, header=False)
label_df.to_csv(os.path.join(output_path, 'SMD_test_label_l.csv'), index=False, header=False)

# ✅ NPY 저장
np.save(os.path.join(output_path, 'SMD_train.npy'), train_df.to_numpy(dtype=np.float32))
np.save(os.path.join(output_path, 'SMD_test.npy'), test_df.to_numpy(dtype=np.float32))

# ✅ 라벨: 마지막 열만 추출하고 NaN은 0으로 처리
label_array = label_df.iloc[:, 0].to_numpy(dtype=np.float32)
label_array = np.nan_to_num(label_array, nan=0.0)
np.save(os.path.join(output_path, 'SMD_test_label.npy'), label_array)

# ✅ 확인 출력
print("✅ 저장 완료:")
print(" - train:", train_df.shape)
print(" - test:", test_df.shape)
print(" - label:", label_array.shape)
print("🧪 label distribution:", np.unique(label_array, return_counts=True))
