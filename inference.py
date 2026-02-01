import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision.datasets import CIFAR10
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter


def run_evaluation():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = "openai/clip-vit-large-patch14"

    print(f"正在加载模型: {model_id}...")
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    # 2. 数据准备
    dataset = CIFAR10(root='./data', train=False, download=True)
    target_map = {3: 0, 5: 1, 2: 2}  # cat:0, dog:1, bird:2
    classes_of_interest = [3, 5, 2]
    filtered_data = [(img, target_map[label]) for img, label in dataset if label in classes_of_interest]

    # 3. 固定 Prompt
    text_prompts = ['a photo of a cat', 'a photo of a dog', 'a photo of a bird']

    # 4. 提取文本特征
    print("提取文本特征...")
    with torch.no_grad():
        inputs_text = processor(text=text_prompts, return_tensors="pt", padding=True).to(device)
        outputs_text = model.get_text_features(**inputs_text)

        # 核心修复：确保获取的是 Tensor
        if not isinstance(outputs_text, torch.Tensor):
            text_features = outputs_text.pooler_output
        else:
            text_features = outputs_text

        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    # 5. 推理循环
    all_preds = []
    all_labels = []

    print("开始高精度推理 (双重增强策略)...")
    for img, label in tqdm(filtered_data):
        with torch.no_grad():
            # 针对 32x32 小图的增强：原始 + 锐化
            # 锐化能显著提升 Large 模型对猫胡须和耳朵轮廓的识别
            img_sharp = img.filter(ImageFilter.SHARPEN)
            imgs = [img, img_sharp]

            # 使用最高质量的 BICUBIC 插值
            inputs = processor(images=imgs, return_tensors="pt").to(device)
            outputs_image = model.get_image_features(**inputs)

            # 核心修复：确保获取的是 Tensor
            if not isinstance(outputs_image, torch.Tensor):
                image_features = outputs_image.pooler_output
            else:
                image_features = outputs_image

            image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

            # 计算相似度
            logit_scale = model.logit_scale.exp()
            logits = (image_features @ text_features.t()) * logit_scale

            # 对两个增强版本的概率进行平均，压制噪声
            probs = torch.softmax(logits, dim=1).mean(dim=0)

            pred = torch.argmax(probs).item()
            all_preds.append(pred)
            all_labels.append(label)

    # 6. 结果统计
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    class_names = ['cat', 'dog', 'bird']

    print("\n" + "=" * 35)
    print(f"{'Class':<12} | {'Accuracy':<10}")
    print("-" * 35)
    for i, name in enumerate(class_names):
        idx = (all_labels == i)
        acc = (all_preds[idx] == all_labels[idx]).mean()
        print(f"{name:<12} | {acc:.2%}")

    print("-" * 35)
    overall_acc = (all_preds == all_labels).mean()
    print(f"{'Overall':<12} | {overall_acc:.2%}")
    print("=" * 35)


if __name__ == "__main__":
    run_evaluation()