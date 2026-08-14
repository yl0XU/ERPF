# ERPF

## Dataset
The dataset used in this project can be downloaded from Baidu Netdisk.

- **Baidu Netdisk Link:**  
  `https://pan.baidu.com/s/1JnrSko_9-ROrc4efFQwE2Q?pwd=y7ec`
- **Extraction Code:**  
  `y7ec`

After downloading and extracting, place the dataset in the following directory: `datasets/`

---

## Pre-trained Model
Pre-trained model weights are available through Baidu Netdisk.

- **Baidu Netdisk Link:**  
  `https://pan.baidu.com/s/1Q8yjuCnMGOkxgd6uYmMQRw?pwd=wyg6`
- **Extraction Code:**  
  `wyg6`

After downloading, place the weights in: `checkpoints/pretrain/`

---

## Trained Model
The trained model weights can also be downloaded from Baidu Netdisk.

- **Baidu Netdisk Link:**  
  `https://pan.baidu.com/s/1PfrP1D_jgIXCOeHZOEx_JA?pwd=7xmi`
- **Extraction Code:**  
  `7xmi`

After downloading, place the weights in: `checkpoints/`

---

## Testing
Once the dataset and model weights are placed in their respective directories, run the following command to test the model:
```bash
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py \
  --config-file "configs/e2e_relation_R_50_FPN.yaml" \
    TEST.IMS_PER_BATCH 1 \
  DTYPE "float32" \
  MODEL.PRETRAINED_DETECTOR_CKPT "checkpoints/model_final.pth " \
  OUTPUT_DIR "your path"
```