import os
from xml.dom.minidom import Document
from pathlib import Path
import cv2

# ====================== 配置区 ======================
# 顶层总目录，程序会递归遍历下面所有子文件夹
ROOT_DIR = r"H:\datasets\test_data\data\F3F50\video\Annotations"
# ====================================================


def create_empty_voc_xml(img_path: Path, save_xml_path: Path):
    """
    生成无object空VOC XML，完全沿用你提供的DOM构建代码
    自动读取图片宽、高、通道数
    """
    img_path_str = str(img_path)
    img = cv2.imread(img_path_str)
    h, w = img.shape[:2]
    depth = img.shape[2] if len(img.shape) == 3 else 1

    folder_name = img_path.parent.name
    img_filename = img_path.name

    # ========== 严格保留你原始的XML构建代码 ==========
    xmlBuilder = Document()
    annotation = xmlBuilder.createElement("annotation")
    xmlBuilder.appendChild(annotation)

    # folder
    folder_elem = xmlBuilder.createElement("folder")
    folder_elem.appendChild(xmlBuilder.createTextNode(folder_name))
    annotation.appendChild(folder_elem)

    # filename
    filename_elem = xmlBuilder.createElement("filename")
    filename_elem.appendChild(xmlBuilder.createTextNode(img_filename))
    annotation.appendChild(filename_elem)

    # path
    path_elem = xmlBuilder.createElement("path")
    path_elem.appendChild(xmlBuilder.createTextNode(img_path_str))
    annotation.appendChild(path_elem)

    # source
    source_elem = xmlBuilder.createElement("source")
    db_elem = xmlBuilder.createElement("database")
    db_elem.appendChild(xmlBuilder.createTextNode("Unknown"))
    source_elem.appendChild(db_elem)
    annotation.appendChild(source_elem)

    # size
    size_elem = xmlBuilder.createElement("size")
    width_elem = xmlBuilder.createElement("width")
    width_elem.appendChild(xmlBuilder.createTextNode(str(w)))
    height_elem = xmlBuilder.createElement("height")
    height_elem.appendChild(xmlBuilder.createTextNode(str(h)))
    depth_elem = xmlBuilder.createElement("depth")
    depth_elem.appendChild(xmlBuilder.createTextNode(str(depth)))
    size_elem.appendChild(width_elem)
    size_elem.appendChild(height_elem)
    size_elem.appendChild(depth_elem)
    annotation.appendChild(size_elem)

    # segmented
    seg_elem = xmlBuilder.createElement("segmented")
    seg_elem.appendChild(xmlBuilder.createTextNode("0"))
    annotation.appendChild(seg_elem)

    # 格式化XML，去掉头部声明
    xml_text = xmlBuilder.toprettyxml(indent="    ", encoding="utf-8").decode("utf-8")
    lines = xml_text.splitlines()[1:]
    xml_clean = "\n".join(lines)

    with open(save_xml_path, "w", encoding="utf-8") as f:
        f.write(xml_clean)


def batch_fill_missing_xml(root_folder):
    root_path = Path(root_folder)
    if not root_path.is_dir():
        print(f"❌ 目录不存在：{root_folder}")
        return

    total_jpg = 0
    created_xml = 0
    skip_exist = 0

    # 递归遍历所有子目录里的jpg图片
    for img_file in root_path.rglob("*.jpg"):
        total_jpg += 1
        # 同名xml文件
        xml_file = img_file.with_suffix(".xml")

        if xml_file.exists():
            skip_exist += 1
            continue

        # 缺失xml，新建空标注文件
        create_empty_voc_xml(img_file, xml_file)
        created_xml += 1
        print(f"✅ 新建：{xml_file}")

    # 输出统计信息
    print("\n" + "=" * 60)
    print(f"📊 汇总统计")
    print(f"总共找到JPG图片：{total_jpg}")
    print(f"已有XML，跳过：{skip_exist}")
    print(f"自动补全空XML文件：{created_xml}")
    print("=" * 60)


if __name__ == "__main__":
    batch_fill_missing_xml(ROOT_DIR)