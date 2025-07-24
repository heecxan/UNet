import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import numpy as np
from pycocotools.coco import COCO

class COCO_Dataset(Dataset):
    def __init__(self, image_dir, ann_file, image_size=256):
        # COCO annotation 파일 로드 (json 파일 파싱)
        self.coco = COCO(ann_file)
        # 이미지 파일이 있는 폴더 경로
        self.image_dir = image_dir
        # 모든 이미지들의 고유 ID 목록 저장
        self.image_ids = list(self.coco.imgs.keys())
        # 원하는 이미지 사이즈
        self.image_size = image_size
        # 이미지 전처리: 리사이즈 + 텐서 변환
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])
        coco_cats = self.coco.loadCats(self.coco.getCatIds())
        self.cat_id_to_index = {cat["id"]: idx for idx, cat in enumerate(coco_cats)}

    def __len__(self):
        # 전체 이미지 개수 반환환
        return len(self.image_ids)

    def __getitem__(self, idx):
        # idx번째 이미지의 고유 ID 얻기
        image_id = self.image_ids[idx]
        # 해당 이미지의 상세 정보 (file_name, height, width 등)
        image_info = self.coco.loadImgs(image_id)[0]
        # 실제 이미지 파일 경로 구성성
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        # 이미지 로드 및 RGB 포멧으로 변환환
        image = Image.open(image_path).convert('RGB')
        # 전처리 적용: 리사이즈 + 텐서 변환환
        image = self.image_transform(image)
        # 해당 이미지에 연결된 annotation ID들 얻기기
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        # annotation ID들을 통해 실제 객체 정보 가져오기
        anns = self.coco.loadAnns(ann_ids)  

        # 빈 마스크 생성 (height x width 크기, 모든 픽셀 0으로 초기화)
        mask = np.zeros((image_info['height'],image_info['width']), dtype=np.uint8)

        # 모든 객체(annotation)마다 마스크 생성
        for ann in anns:
            coco_cat_id = ann['category_id']
            mapped_cat = self.cat_id_to_index[coco_cat_id]
            mask_instance = self.coco.annToMask(ann)
            mask[mask_instance == 1] = mapped_cat

        # 넘파이 배열 → PIL 이미지로 변환 (리사이즈 위해)
        mask = Image.fromarray(mask)
        # 마스크도 리사이즈 (interpolation=NEAREST → 클래스 값 보존)
        mask = T.Resize((self.image_size, self.image_size), interpolation=Image.NEAREST)(mask)
        # 다시 텐서로 변환 (정수형, segmentation에서 필요)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask
