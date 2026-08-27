import os
import shutil
from pathlib import Path
import cv2
from xml.dom.minidom import Document


def copy_or_create_xml(folder_list: list[str], search_root: str):
    """
    :param folder_list: 待处理根文件夹列表，每个根下有images目录
    :param search_root: 源xml总查找目录（递归搜所有*.xml）
    """
    img_suffix = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    search_root_path = Path(search_root)

    if not search_root_path.exists():
        print(f"【警告】源xml查找目录不存在: {search_root}")
        return

    # 预缓存所有xml文件名映射 stem -> xml路径
    xml_map = {}
    for xml_file in search_root_path.rglob("*.xml"):
        stem = xml_file.stem
        xml_map[stem] = xml_file

    for root_dir in folder_list:
        root_path = Path(root_dir)
        images_dir = root_path / "images"
        annotations_dir = root_path / "Annotations"

        if not images_dir.exists():
            print(f"\n【跳过】不存在images目录: {images_dir}")
            continue
        annotations_dir.mkdir(exist_ok=True)
        folder_name = images_dir.name  # images / images3 ...
        print(f"\n===== 处理目录: {root_dir} =====")

        for img_file in images_dir.iterdir():
            # 跳过子文件夹、隐藏文件
            if img_file.is_dir() or img_file.name.startswith("."):
                continue
            suf = img_file.suffix.lower()
            if suf not in img_suffix:
                continue

            stem = img_file.stem
            target_xml = annotations_dir / f"{stem}.xml"

            if stem in xml_map:
                # 存在源xml，直接复制
                src_xml = xml_map[stem]
                shutil.copy2(src_xml, target_xml)
                print(f"已复制: {src_xml.name} -> {target_xml.name}")
            else:
                # 无匹配xml，生成标准VOC空白标注xml，使用opencv读取尺寸
                img_path = str(img_file)
                w, h, depth = 1632, 1224, 3
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        # 判断通道数
                        if len(img.shape) == 3:
                            depth = img.shape[2]
                        else:
                            depth = 1
                    else:
                        raise Exception("cv2读取图像返回None")
                except Exception as e:
                    print(f"读取图片失败 {img_file.name}: {e}，使用默认尺寸1632x1224 depth3")

                # ========== 使用xml.dom.minidom构建XML ==========
                xmlBuilder = Document()
                annotation = xmlBuilder.createElement("annotation")
                xmlBuilder.appendChild(annotation)

                # folder
                folder_elem = xmlBuilder.createElement("folder")
                folder_elem.appendChild(xmlBuilder.createTextNode(folder_name))
                annotation.appendChild(folder_elem)

                # filename
                filename_elem = xmlBuilder.createElement("filename")
                filename_elem.appendChild(xmlBuilder.createTextNode(img_file.name))
                annotation.appendChild(filename_elem)

                # path
                path_elem = xmlBuilder.createElement("path")
                path_elem.appendChild(xmlBuilder.createTextNode(str(img_file)))
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

                # 生成带缩进、换行的格式化xml字符串
                xml_text = xmlBuilder.toprettyxml(indent="    ", encoding="utf-8").decode("utf-8")
                # 去掉默认第一行 <?xml ...?> 声明（和你提供的样例保持一致）
                lines = xml_text.splitlines()[1:]
                xml_clean = "\n".join(lines)

                with open(target_xml, "w", encoding="utf-8") as f:
                    f.write(xml_clean)
                print(f"未找到源xml，新建标注文件: {target_xml.name}")


def main():
    # 配置区自行修改
    path_list = [
        r"H:\datasets\Basic_Single_Fish\T07\D1\S0\A0\W0\L1\20260721_F46",
        r"H:\datasets\Basic_Single_Fish\T07\D1\S0\A0\W0\L2\20260721_F46",
        r"H:\datasets\Basic_Single_Fish\T07\D1\S0\A0\W0\L3\20260721_F46",
        r"H:\datasets\Basic_Single_Fish\T07\D1\S0\A0\W0\L4\20260721_F46",
        r"H:\datasets\Basic_Single_Fish\T07\D1\S0\A0\W0\L5\20260721_F46",
        r"H:\datasets\Basic_Single_Fish\T07\D1\S0\A0\W0\L6\20260721_F46",
        r"H:\datasets\Basic_Single_Fish\T07\D1\S0\A0\W0\L7\20260721_F46",
        r"H:\datasets\Basic_Single_Fish\T07\D1\S0\A0\W0\L8\20260721_F46"
    ]
    xml_search_dir = r"H:\datasets\Basic_Single_Fish\T07\D1\S0\A0\W0\F46\Annotations"  # 存放原始标注xml的总目录

    copy_or_create_xml(path_list, xml_search_dir)


if __name__ == "__main__":
    main()