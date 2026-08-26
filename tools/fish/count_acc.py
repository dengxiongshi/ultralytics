import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ===================== 配置参数 =====================
MODEL_PATH = r"D:\python_work\ultralytics\runs\detect\fish\yolov8n\weights\best.pt"
IMG_DIR = r"H:\datasets\test_data\data\F50\video\Annotations\2026061217_1781258299"
XML_DIR = IMG_DIR
SAVE_VIDEO_DIR = r"H:\datasets\test_data\data\F50\video\Annotations"
os.makedirs(SAVE_VIDEO_DIR, exist_ok=True)

IOU_THRESH = 0.45
CONF_THRESH = 0.25
IMGSZ = 640
FPS = 15

# 中文字体路径
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
FONT_SIZE_LARGE = 28
FONT_SIZE_SMALL = 18

# 类别映射
CLASS_NAME_TO_ID = {
    "Danio rerio": 0,
    "Pterophyllum scalare": 1,
}
ENG_TO_CN = {
    "Danio rerio": "斑马",
    "Pterophyllum scalare": "天使",
}
ID_TO_CLASS = {
    cls_id: ENG_TO_CN[eng_name]
    for eng_name, cls_id in CLASS_NAME_TO_ID.items()
}
NUM_CLASSES = len(CLASS_NAME_TO_ID)
# ===================================================


def parse_xml(xml_path):
    targets = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.iter("object"):
            name_node = obj.find("name")
            bbox_node = obj.find("bndbox")
            if name_node is None or bbox_node is None:
                continue
            cls_name = name_node.text.strip()
            if not cls_name:
                continue
            try:
                x1 = float(bbox_node.find("xmin").text)
                y1 = float(bbox_node.find("ymin").text)
                x2 = float(bbox_node.find("xmax").text)
                y2 = float(bbox_node.find("ymax").text)
                targets.append((cls_name, int(x1), int(y1), int(x2), int(y2)))
            except (ValueError, TypeError):
                continue
    except ET.ParseError:
        print(f"警告：{xml_path} 解析失败，按空目标处理")
    return targets


def calculate_iou(box1, box2):
    """计算单个box与单个box的IOU，与ultralytics逻辑一致"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def match_predictions(gt_targets, pred_boxes, pred_cls_ids, pred_confs):
    """
    对齐 YOLOv8 官方验证匹配逻辑：
    1. 按类别拆分真值与预测
    2. 预测按置信度降序排序
    3. 同类别内贪心IOU匹配，每个真值仅匹配一次
    返回：单帧TP数、真值总数、预测总数、单帧召回率、单帧精确率、单帧F1
    """
    gt_count = len(gt_targets)
    pred_count = len(pred_boxes)

    # 边界场景直接返回
    if gt_count == 0 and pred_count == 0:
        return 0, 0, 0, 1.0, 1.0, 1.0
    if gt_count == 0 and pred_count > 0:
        return 0, 0, pred_count, 0.0, 0.0, 0.0
    if gt_count > 0 and pred_count == 0:
        return 0, gt_count, 0, 0.0, 0.0, 0.0

    # 按类别分组真值，和官方逻辑一致
    gt_by_class = [[] for _ in range(NUM_CLASSES)]
    for cls_name, x1, y1, x2, y2 in gt_targets:
        cls_id = CLASS_NAME_TO_ID.get(cls_name, -1)
        if cls_id >= 0:
            gt_by_class[cls_id].append([x1, y1, x2, y2])

    tp = 0  # 单帧真正例总数

    # 按类别逐个匹配，和官方val逐类别计算逻辑对齐
    for cls_id in range(NUM_CLASSES):
        cls_gts = gt_by_class[cls_id]
        if not cls_gts:
            continue

        # 提取当前类别的所有预测，按置信度降序排序
        cls_pred_mask = pred_cls_ids == cls_id
        cls_pred_boxes = pred_boxes[cls_pred_mask]
        cls_pred_confs = pred_confs[cls_pred_mask]
        if len(cls_pred_boxes) == 0:
            continue

        # 置信度降序
        sort_idx = np.argsort(-cls_pred_confs)
        cls_pred_boxes = cls_pred_boxes[sort_idx]

        matched_gt = set()

        # 逐个预测框匹配最佳真值
        for pred_box in cls_pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(cls_gts):
                if gt_idx in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= IOU_THRESH and best_gt_idx != -1:
                matched_gt.add(best_gt_idx)
                tp += 1

    # 单帧指标
    recall = tp / gt_count if gt_count > 0 else 0.0
    precision = tp / pred_count if pred_count > 0 else 0.0

    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * recall * precision / (recall + precision)

    return tp, gt_count, pred_count, recall, precision, f1


def draw_info_and_box(img_bgr, gt_num, pred_num, recall, precision, f1,
                      pred_boxes, pred_cls_ids, pred_confs, gt_targets):
    # 第一步：OpenCV画矩形框
    for (eng_cls, x1, y1, x2, y2) in gt_targets:
        cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 第二步：PIL画中文标签
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    try:
        font_large = ImageFont.truetype(FONT_PATH, FONT_SIZE_LARGE)
        font_small = ImageFont.truetype(FONT_PATH, FONT_SIZE_SMALL)
    except IOError:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # 顶部统计文字
    top_text = (f"GT:{gt_num}  PRED:{pred_num}  "
                f"召回:{recall*100:.1f}%  精确:{precision*100:.1f}%  F1:{f1*100:.1f}%")
    draw.text((10, 10), top_text, font=font_large, fill=(255, 0, 0))

    # 真值标签
    for (eng_cls, x1, y1, x2, y2) in gt_targets:
        cn_cls = ENG_TO_CN.get(eng_cls, "未知")
        label = f"GT:{cn_cls}"
        label_y = int(y2) + 2
        text_bbox = draw.textbbox((int(x1), label_y), label, font=font_small)
        draw.rectangle(text_bbox, fill=(0, 255, 0))
        draw.text((int(x1), label_y), label, font=font_small, fill=(0, 0, 0))

    # 预测标签
    for box, cls_id, conf in zip(pred_boxes, pred_cls_ids, pred_confs):
        x1, y1, x2, y2 = map(int, box)
        cls_name = ID_TO_CLASS.get(cls_id, "未知类别")
        label = f"{cls_name} {conf:.2f}"
        label_y = y1 - FONT_SIZE_SMALL - 4
        text_bbox = draw.textbbox((x1, label_y), label, font=font_small)
        draw.rectangle(text_bbox, fill=(255, 0, 0))
        draw.text((x1, label_y), label, font=font_small, fill=(255, 255, 255))

    img_result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img_result


def images_to_video(image_list, save_path, fps=15):
    if not image_list:
        return
    h, w = image_list[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for frame in image_list:
        writer.write(frame)
    writer.release()
    print(f"视频已保存: {save_path}")


def main():
    model = YOLO(MODEL_PATH)
    print(f"模型加载完成：{MODEL_PATH}\n")

    img_files = [f for f in os.listdir(IMG_DIR)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"找到测试图片：{len(img_files)} 张\n")

    record_list = []
    empty_gt_count = 0

    # ========== 新增：全局统计变量（对齐YOLOv8官方口径） ==========
    total_tp = 0          # 全数据集总正确匹配数
    total_gt = 0          # 全数据集总真值数
    total_pred = 0        # 全数据集总预测数
    total_false_positive = 0

    for img_name in img_files:
        img_full_path = os.path.join(IMG_DIR, img_name)
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_path = os.path.join(XML_DIR, xml_name)

        gt_targets = parse_xml(xml_path) if os.path.exists(xml_path) else []
        gt_num = len(gt_targets)
        if gt_num == 0:
            empty_gt_count += 1

        try:
            result = model.predict(
                source=img_full_path,
                conf=CONF_THRESH,
                imgsz=IMGSZ,
                verbose=False
            )[0]
        except Exception as e:
            print(f"推理失败 {img_name}: {e}")
            continue

        if result.boxes is None:
            pred_boxes = np.empty((0, 4))
            pred_cls = np.array([], dtype=int)
            pred_conf = np.array([])
        else:
            pred_boxes = result.boxes.xyxy.cpu().numpy()
            pred_cls = result.boxes.cls.cpu().numpy().astype(int)
            pred_conf = result.boxes.conf.cpu().numpy()
        pred_num = len(pred_boxes)

        tp, gt_count, pred_count, recall, precision, f1 = match_predictions(
            gt_targets, pred_boxes, pred_cls, pred_conf
        )

        # 累加到全局统计
        total_tp += tp
        total_gt += gt_count
        total_pred += pred_count
        total_false_positive += (pred_count - tp)

        # 读图+画框+写字
        img_bgr = cv2.imread(img_full_path)
        img_bgr = draw_info_and_box(
            img_bgr, gt_num, pred_num, recall, precision, f1,
            pred_boxes, pred_cls, pred_conf, gt_targets
        )

        record_list.append((
            img_full_path, recall*100, precision*100, f1*100,
            gt_num, pred_num, img_bgr
        ))

        print(f"{img_name:22s} | GT:{gt_num:2d} PRED:{pred_num:2d} | "
              f"召回:{recall*100:6.2f}%  精确:{precision*100:6.2f}%  "
              f"F1:{f1*100:6.2f}%  误检:{pred_count-tp}个")

    if not record_list:
        print("无有效数据")
        return

    # ========== 统计计算 ==========
    # 1. 单帧平均口径
    all_recall = [r[1] for r in record_list]
    all_precision = [r[2] for r in record_list]
    all_f1 = [r[3] for r in record_list]

    # 仅含目标帧
    target_records = [r for r in record_list if r[4] > 0]
    if target_records:
        t_recall = [r[1] for r in target_records]
        t_precision = [r[2] for r in target_records]
        t_f1 = [r[3] for r in target_records]
        t_max_f1_idx = np.argmax(t_f1)
        t_min_f1_idx = np.argmin(t_f1)

    # 2. 全局汇总口径（和YOLOv8 val完全一致）
    global_recall = total_tp / total_gt if total_gt > 0 else 0.0
    global_precision = total_tp / total_pred if total_pred > 0 else 0.0
    if global_recall + global_precision == 0:
        global_f1 = 0.0
    else:
        global_f1 = 2 * global_recall * global_precision / (global_recall + global_precision)

    # ========== 打印结果 ==========
    print("\n" + "="*80)
    print("【统计结果汇总】")
    print(f"总图片数：{len(record_list)} 张")
    print(f"空真值图片数：{empty_gt_count} 张（占比 {empty_gt_count/len(record_list)*100:.1f}%）")
    print(f"含目标图片数：{len(target_records)} 张")
    print(f"全局总真值数：{total_gt} 个")
    print(f"全局总预测数：{total_pred} 个")
    print(f"全局正确匹配数：{total_tp} 个")
    print(f"全局总误检数：{total_false_positive} 个")
    print("="*80)

    print("\n【1. 全局统计口径（与 YOLOv8 val 完全对齐）】")
    print(f"  全局召回率：{global_recall*100:.2f}%  （总正确数 / 总真值数）")
    print(f"  全局精确率：{global_precision*100:.2f}%  （总正确数 / 总预测数）")
    print(f"  全局 F1 分数：{global_f1*100:.2f}%")

    print("\n【2. 单帧平均口径（含空帧）】")
    print(f"  平均召回率：{np.mean(all_recall):.2f}%")
    print(f"  平均精确率：{np.mean(all_precision):.2f}%")
    print(f"  平均 F1 分数：{np.mean(all_f1):.2f}%")

    if target_records:
        print("\n【3. 单帧平均口径（排除空帧，真实检测能力）】")
        print(f"  平均召回率：{np.mean(t_recall):.2f}%")
        print(f"  平均精确率：{np.mean(t_precision):.2f}%")
        print(f"  平均 F1 分数：{np.mean(t_f1):.2f}%")
        print("-" * 80)
        print("  【F1 极值（排除空帧）】")
        print(f"  ✅ 最高 F1：{t_f1[t_max_f1_idx]:.2f}%")
        print(f"     对应图片：{target_records[t_max_f1_idx][0]}")
        print(f"  ❌ 最低 F1：{t_f1[t_min_f1_idx]:.2f}%")
        print(f"     对应图片：{target_records[t_min_f1_idx][0]}")
    print("="*80)

    # 生成视频
    all_frames = [item[6] for item in record_list]
    parent_folder_name = os.path.basename(IMG_DIR)
    video_name = f"{parent_folder_name}.mp4"
    video_save_path = os.path.join(SAVE_VIDEO_DIR, video_name)
    images_to_video(all_frames, video_save_path, fps=FPS)


if __name__ == "__main__":
    main()