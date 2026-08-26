import os


def rename_mp4_in_subdirs(root_dir):
    """
    递归遍历所有子目录，批量重命名MP4文件
    命名规则：子目录名_原文件下划线后的部分.mp4
    """
    # 递归遍历所有子目录
    for dirpath, _, filenames in os.walk(root_dir):
        # 获取当前子目录的名称（重命名的核心前缀）
        dir_name = os.path.basename(dirpath)

        # 遍历当前目录下的所有文件
        for filename in filenames:
            # 只处理MP4文件（忽略大小写）
            if not filename.lower().endswith(".mp4"):
                continue

            # 分割文件名：仅分割1次下划线，提取后半部分
            # 格式：前缀_数字.mp4 → 提取 数字.mp4
            if "_" in filename:
                # 分割一次，保留后面所有内容（适配多下划线场景）
                _, file_suffix = filename.split("_", 1)
                # 拼接新文件名
                new_filename = f"{dir_name}_{file_suffix}"
            else:
                # 无下划线的MP4，直接前缀+原名
                new_filename = f"{dir_name}_{filename}"

            # 拼接完整路径
            old_path = os.path.join(dirpath, filename)
            new_path = os.path.join(dirpath, new_filename)

            # 避免重名覆盖
            if os.path.exists(new_path):
                print(f"⚠️  已存在，跳过：{new_filename}")
                continue

            # 执行重命名
            try:
                os.rename(old_path, new_path)
                print(f"✅ 重命名成功：{filename} → {new_filename}")
            except Exception as e:
                print(f"❌ 重命名失败 {filename}：{str(e)}")


if __name__ == "__main__":
    # ========== 请修改为你的顶层目录路径 ==========
    ROOT_FOLDER = r"H:\datasets\5\20260612_F3"
    # ==============================================

    if not os.path.exists(ROOT_FOLDER):
        print("❌ 错误：根目录不存在，请检查路径！")
    else:
        rename_mp4_in_subdirs(ROOT_FOLDER)
        print("\n🎉 所有MP4文件重命名完成！")