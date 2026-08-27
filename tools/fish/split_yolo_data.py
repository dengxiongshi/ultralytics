import os
import shutil
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path

# ===================== 配置区（自行修改） =====================
# 1. 多个源数据集文件夹，每个下有 images、Annotations
path_list = [
    r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W0\L3\20260610_F50",
    r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W1\L3\20260611_F50",
    r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W2\L3\20260612_F50",
    r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W1\L3\20260612_F3",
    r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W2\L3\20260612_F3",
    r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W1\L3\20260611_F3F50",
    r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W2\L3\20260612_F3F50",
    r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W0\L4\20260610_F50",
    r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W1\L4\20260611_F50",
    r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W2\L4\20260612_F50",
    r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W1\L4\20260612_F3",
    r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W2\L4\20260612_F3",
    r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W1\L4\20260611_F3F50",
    r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W2\L4\20260612_F3F50",
    r"H:\datasets\Basic_Single_Fish\T13\D1\S0\A1\W0\L3\20260610_F3",
    r"H:\datasets\Basic_Single_Fish\T13\D1\S0\A1\W0\L3\20260611_F3F50",
    r"H:\datasets\Basic_Single_Fish\T13\D1\S0\A1\W0\L4\20260610_F3",
    r"H:\datasets\Basic_Single_Fish\T13\D1\S0\A1\W0\L4\20260611_F3F50"
]
# 2. 输出YOLO数据集根目录
OUTPUT_ROOT = r"H:\datasets\train_data_20260826"
# 3. 划分比例 train:val:test = 7:1.5:1.5（仅样本>50时生效）
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
# 4. 图片支持后缀
IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
# 5. 阈值：大于该数量才进行三集划分，否则全部进train
SPLIT_THRESHOLD = 50

# ===================== 关键：自定义固定类别列表 =====================
# 在这里手动写死所有类别，顺序对应label id 0,1,2...
# 如果置为空列表，则自动从xml里提取并排序类别
CUSTOM_CLASS_NAMES = ["斑马", "天使"]
# ===================================================================


def create_dirs(root: Path):
    dirs = [
        root / "images" / "train",
        root / "images" / "val",
        root / "images" / "test",
        root / "labels" / "train",
        root / "labels" / "val",
        root / "labels" / "test",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_all_classes(all_xml_list: list) -> list:
    """
    获取类别：
    1. 如果CUSTOM_CLASS_NAMES不为空，直接使用自定义列表
    2. 为空则自动采集所有xml类别并排序
    """
    if len(CUSTOM_CLASS_NAMES) > 0:
        print("使用用户自定义固定类别列表")
        return CUSTOM_CLASS_NAMES

    temp_cls = set()
    for xml_path in all_xml_list:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip()
            temp_cls.add(cls_name)
    class_names = sorted(list(temp_cls))
    print("未配置自定义类别，自动从数据集提取类别")
    return class_names


def parse_voc_xml(xml_path: Path, class_names: list) -> list:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    yolo_anns = []
    objs = root.findall("object")
    if not objs:
        return yolo_anns

    unknown_cls_set = set()
    for obj in objs:
        cls_name = obj.find("name").text.strip()
        if cls_name not in class_names:
            unknown_cls_set.add(cls_name)
            continue
        cls_id = class_names.index(cls_name)

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        yolo_anns.append((cls_id, x_center, y_center, w, h))

    # 单个xml只打印一次警告
    if unknown_cls_set:
        abs_xml_path = xml_path.resolve()
        print(f"警告：文件 [{abs_xml_path}] 包含未定义类别：{list(unknown_cls_set)}")
    return yolo_anns


def split_single_folder(file_list: list):
    """单套文件夹内部顺序截断：前70%train，中间15%val，末尾15%test"""
    total = len(file_list)
    train_end_idx = int(total * TRAIN_RATIO)
    val_end_idx = train_end_idx + int(total * VAL_RATIO)

    train = file_list[:train_end_idx]
    val = file_list[train_end_idx:val_end_idx]
    test = file_list[val_end_idx:]
    return train, val, test


def generate_yaml(root: Path, class_names: list):
    """生成YOLO训练用dataset.yaml，names为id:name字典格式"""
    names_dict = {i: name for i, name in enumerate(class_names)}
    yaml_data = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": names_dict
    }
    yaml_path = root / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=None, allow_unicode=True, indent=2)
    print(f"\n✅ YOLO数据集配置文件已生成：{yaml_path}")


def main():
    output_root = Path(OUTPUT_ROOT)
    create_dirs(output_root)

    # 全局汇总容器
    global_train = []
    global_val = []
    global_test = []
    all_xml_paths = []

    for src_dir in path_list:
        src = Path(src_dir)
        img_dir = src / "images"
        ann_dir = src / "Annotations"
        if not img_dir.exists() or not ann_dir.exists():
            print(f"⚠️ 警告：{src_dir} 缺少 images/Annotations，跳过")
            continue

        # 收集当前文件夹下所有图片+xml配对
        folder_pairs = []
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() not in IMG_SUFFIXES:
                continue
            xml_path = ann_dir / f"{img_path.stem}.xml"
            if not xml_path.exists():
                print(f"⚠️ 警告：图片 {img_path.name} 无对应xml，跳过")
                continue
            folder_pairs.append((img_path, xml_path))
            all_xml_paths.append(xml_path)

        folder_total = len(folder_pairs)
        if folder_total == 0:
            print(f"文件夹 {src_dir} 无有效数据，跳过")
            continue

        # ============ 新增分支逻辑 ============
        if folder_total > SPLIT_THRESHOLD:
            folder_train, folder_val, folder_test = split_single_folder(folder_pairs)
            print(f"\n【子数据集】{src.name} 总量{folder_total}(>{SPLIT_THRESHOLD},执行划分) | train:{len(folder_train)} val:{len(folder_val)} test:{len(folder_test)}")
        else:
            folder_train = folder_pairs
            folder_val = []
            folder_test = []
            print(f"\n【子数据集】{src.name} 总量{folder_total}(≤{SPLIT_THRESHOLD},全部归入train集)")

        # 追加到全局集合
        global_train.extend(folder_train)
        global_val.extend(folder_val)
        global_test.extend(folder_test)

    # 汇总全部数据
    total_all = len(global_train) + len(global_val) + len(global_test)
    print(f"\n📊 全局汇总有效图片标注对：{total_all}")
    print(f"全局train: {len(global_train)} 张")
    print(f"全局val:   {len(global_val)} 张")
    print(f"全局test:  {len(global_test)} 张")
    if total_all == 0:
        print("❌ 无数据，程序退出")
        return

    # 获取固定类别映射
    class_names = get_all_classes(all_xml_paths)
    print(f"📋 类别列表 {class_names}，类别总数：{len(class_names)}")

    empty_counter = {
        "train": 0,
        "val": 0,
        "test": 0
    }
    # 批量处理三个集合
    split_map = {
        "train": global_train,
        "val": global_val,
        "test": global_test
    }
    for split_name, pair_list in split_map.items():
        if len(pair_list) == 0:
            continue
        print(f"\n===== 开始处理 {split_name} 集（{len(pair_list)}张） =====")
        dst_img = output_root / "images" / split_name
        dst_label = output_root / "labels" / split_name

        for img_path, xml_path in pair_list:
            # 复制原图
            shutil.copy2(img_path, dst_img / img_path.name)
            # xml转yolo txt
            anns = parse_voc_xml(xml_path, class_names)
            txt_file = dst_label / f"{img_path.stem}.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                for line in anns:
                    c, xc, yc, w, h = line
                    f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            # 统计空标注
            if len(anns) == 0:
                empty_counter[split_name] += 1
                print(f"空标注：{img_path.stem}")

    total_empty = sum(empty_counter.values())
    print("\n" + "=" * 50)
    print("📋 空标注统计结果：")
    print(f"  train 集空标注数量：{empty_counter['train']}")
    print(f"  val   集空标注数量：{empty_counter['val']}")
    print(f"  test  集空标注数量：{empty_counter['test']}")
    print(f"  全局空标注图片总数：{total_empty}")
    print("=" * 50)

    # 输出classes.txt
    class_file = output_root / "classes.txt"
    with open(class_file, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"\n📄 类别文本文件保存至: {class_file}")

    # 生成YOLO yaml配置文件
    generate_yaml(output_root, class_names)

    print("🎉 数据集转换划分全部完成！")


if __name__ == "__main__":
    main()
