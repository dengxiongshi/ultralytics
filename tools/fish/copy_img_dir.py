import os
import shutil

# ====================== 【仅修改此处配置】 ======================
# 顶层根目录（会递归扫描所有子文件夹）
ROOT_SCAN_PATH = r"H:\datasets\Basic_Single_Fish"
# 图片统一存放的目标路径
DEST_SAVE_PATH = r"H:\datasets\all_fish_images"
# 支持的图片后缀（大小写自动兼容）
IMG_SUFFIX = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")
# =================================================================

def get_unique_file_path(target_dir: str, file_name: str) -> str:
    """
    处理重名文件，生成不重复的保存路径
    例：a.jpg → a_1.jpg / a_2.jpg
    """
    save_path = os.path.join(target_dir, file_name)
    if not os.path.exists(save_path):
        return save_path

    name, ext = os.path.splitext(file_name)
    count = 1
    while True:
        new_name = f"{name}_{count}{ext}"
        new_path = os.path.join(target_dir, new_name)
        if not os.path.exists(new_path):
            return new_path
        count += 1

def copy_all_images_from_images_folder(root_dir: str, dest_dir: str):
    # 创建目标文件夹，不存在则新建
    os.makedirs(dest_dir, exist_ok=True)
    total_copy = 0
    skip_error = 0

    print("=" * 65)
    print(f"🚀 开始递归扫描目录：{root_dir}")
    print(f"📂 所有images内图片统一拷贝至：{dest_dir}")
    print(f"🖼️  支持图片格式：{IMG_SUFFIX}")
    print("=" * 65 + "\n")

    # 递归遍历所有目录
    for dirpath, subdirs, filenames in os.walk(root_dir):
        # 判断当前文件夹是否为 images
        if os.path.basename(dirpath) == "images":
            print(f"🔍 找到images目录：{dirpath}")
            for file in filenames:
                file_lower = file.lower()
                # 仅处理图片文件
                if file_lower.endswith(IMG_SUFFIX):
                    src_file = os.path.join(dirpath, file)
                    try:
                        # 获取无冲突的目标路径
                        dst_file = get_unique_file_path(dest_dir, file)
                        shutil.copy2(src_file, dst_file)
                        print(f"✅ 拷贝成功：{os.path.basename(src_file)} -> {os.path.basename(dst_file)}")
                        total_copy += 1
                    except Exception as e:
                        print(f"❌ 拷贝失败 {src_file}，原因：{str(e)}")
                        skip_error += 1
            print()

    # 任务完成统计
    print("=" * 65)
    print("🎉 批量拷贝任务结束！")
    print(f"✅ 成功拷贝图片总数：{total_copy}")
    print(f"❌ 拷贝失败跳过文件数：{skip_error}")
    print("=" * 65)


if __name__ == "__main__":
    copy_all_images_from_images_folder(ROOT_SCAN_PATH, DEST_SAVE_PATH)