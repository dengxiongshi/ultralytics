import os
import re
import xml.etree.ElementTree as ET

# 隐藏的0xFEFF BOM字符
FEFF_CHAR = chr(0xFEFF)

def clean_all_xml_in_dir(xml_root_dir: str, overwrite: bool = False):
    """
    遍历指定文件夹下所有xml，清除<name>节点内的0xFEFF不可见字符
    :param xml_root_dir: 存放xml文件的文件夹路径
    :param overwrite: True=直接覆盖原xml；False=生成xxx_clean.xml新文件（默认安全模式）
    """
    if not os.path.isdir(xml_root_dir):
        print(f"错误：路径不存在或不是文件夹 -> {xml_root_dir}")
        return

    # 递归遍历所有xml文件
    file_count = 0
    clean_count = 0
    for root, _, files in os.walk(xml_root_dir):
        for filename in files:
            if filename.lower().endswith(".xml"):
                file_count += 1
                file_path = os.path.join(root, filename)
                try:
                    # 解析xml
                    tree = ET.parse(file_path)
                    root_node = tree.getroot()
                    modified = False

                    # 遍历全部<name>节点，清除FEFF字符
                    for name_node in root_node.findall(".//name"):
                        if name_node.text and FEFF_CHAR in name_node.text:
                            name_node.text = name_node.text.replace(FEFF_CHAR, "")
                            modified = True

                    # 文件无修改则跳过保存
                    if not modified:
                        continue
                    clean_count += 1

                    # 输出路径逻辑
                    if overwrite:
                        save_path = file_path
                    else:
                        name_no_ext, ext = os.path.splitext(file_path)
                        save_path = f"{name_no_ext}_clean{ext}"

                    # 写入，保留xml声明、utf-8编码
                    tree.write(
                        save_path,
                        encoding="utf-8",
                        xml_declaration=True
                    )
                    print(f"已清洗：{os.path.relpath(save_path, xml_root_dir)}")

                except Exception as e:
                    print(f"处理失败 {file_path}，错误：{str(e)}")

    print(f"\n处理完成！共扫描 {file_count} 个XML，成功清洗 {clean_count} 个含异常字符的文件")


def rename_by_regex(folder_path: str, old_tag: str, new_tag: str):
    """
    将文件名中独立的标签 old_tag 替换为 new_tag
    匹配规则：old_tag后面紧跟非字母数字 或者 位于字符串末尾

    Args:
        folder_path: 目标文件夹路径
        old_tag: 需要被替换的字符串，例如 "W1"
        new_tag: 替换后的字符串，例如 "W0"
    """
    # 自动转义元字符，防止 old_tag 含 . * + 等正则符号出错
    escaped_old = re.escape(old_tag)
    pattern = re.compile(rf"{escaped_old}(?=[^a-zA-Z0-9]|$)")

    for filename in os.listdir(folder_path):
        old_full = os.path.join(folder_path, filename)
        if os.path.isdir(old_full):
            continue

        new_filename = pattern.sub(new_tag, filename)
        if new_filename != filename:
            new_full = os.path.join(folder_path, new_filename)
            # 防止重名覆盖
            if os.path.exists(new_full):
                print(f"跳过，目标文件已存在: {new_filename}")
                continue
            os.rename(old_full, new_full)
            print(f"重命名: {filename} --> {new_filename}")


def modify_xml_labels(xml_folder: str):
    """
    遍历文件夹下所有xml，
    Danio rerio → 斑马
    Pterophyllum scalare → 天使
    """
    # 替换字典
    replace_map = {
        "Danio rerio": "斑马",
        "Pterophyllum scalare": "天使"
    }

    for filename in os.listdir(xml_folder):
        if not filename.lower().endswith(".xml"):
            continue
        xml_path = os.path.join(xml_folder, filename)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            changed = False
            # 遍历所有object下的name节点
            for obj in root.findall("object"):
                name_node = obj.find("name")
                if name_node is not None and name_node.text in replace_map:
                    old_text = name_node.text
                    name_node.text = replace_map[old_text]
                    changed = True
                    print(f"[{filename}] {old_text} → {replace_map[old_text]}")
            if changed:
                tree.write(xml_path, encoding="utf-8")
        except Exception as e:
            print(f"处理失败 {xml_path}, 错误: {e}")


if __name__ == "__main__":
    # ====================== 在这里修改你的xml文件夹路径 ======================
    XML_FOLDER_PATH = r"H:\datasets\Basic_Single_Fish\T07\D1\S0\A0\W0\F46\Annotations"  # 存放所有xml的目录
    IS_OVERWRITE = True  # False：生成新文件；True：直接覆盖原文件（谨慎使用）
    # =========================================================================
    clean_all_xml_in_dir(XML_FOLDER_PATH, overwrite=IS_OVERWRITE)

    # target_dir = r"H:\datasets\Basic_Single_Fish\T13\D1\S0\A1\W0\L3\20260610_F3\images"
    # old_tag = "A0"
    # new_tag = "A1"
    # rename_by_regex(target_dir, old_tag=old_tag, new_tag=new_tag)

    # 改成你的xml文件夹路径
    # xml_dir = r"H:\datasets\Basic_Single_Fish\T05\D1\S0\A0\W0\F3F50\Annotations"
    # modify_xml_labels(xml_dir)