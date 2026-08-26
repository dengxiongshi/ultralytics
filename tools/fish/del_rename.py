import os

# ====================== 【全局统一配置区 - 仅修改此处】 ======================
# 顶层根目录（两个功能共用同一个目录）
TARGET_ROOT = r"H:\datasets\11"

# 需要批量删除的文件后缀列表（按需增删）
DELETE_SUFFIX_LIST = [".jpg", ".jpeg", ".png", ".txt", ".gif"]
# =============================================================================


def batch_delete_files(root_dir: str, delete_formats: list):
    """
    递归删除多层子目录中指定后缀的文件
    :param root_dir: 顶层根目录
    :param delete_formats: 要删除的文件后缀列表
    """
    root_path = os.path.abspath(root_dir)
    if not os.path.isdir(root_path):
        print(f"❌ 错误：目录不存在 → {root_path}")
        return

    print("=" * 55)
    print(f"📂 扫描根目录：{root_path}")
    print(f"🗑️  待删除格式：{delete_formats}")
    print("⚠️  文件删除后不可恢复，请谨慎确认！")
    print("=" * 55)

    # 二次安全确认
    if input("\n输入【y】确认执行删除，其他键退出：").strip().lower() != "y":
        print("\n🚫 已取消删除操作")
        return

    success_count = 0
    fail_count = 0

    # 递归遍历所有目录
    for dirpath, _, filenames in os.walk(root_path):
        if not filenames:
            continue

        for filename in filenames:
            file_lower = filename.lower()
            if any(file_lower.endswith(ext.lower()) for ext in delete_formats):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"✅ 已删除：{file_path}")
                    success_count += 1
                except Exception as e:
                    print(f"❌ 删除失败：{file_path} | 原因：{str(e)}")
                    fail_count += 1

    # 统计结果
    print("\n" + "=" * 55)
    print("🎉 批量删除任务执行完成！")
    print(f"✅ 成功删除：{success_count} 个文件")
    print(f"❌ 删除失败：{fail_count} 个文件")
    print("=" * 55 + "\n")


def rename_mp4_in_subdirs(root_dir):
    """
    递归遍历多层子目录，批量重命名MP4文件
    规则：子目录名_原文件第一个下划线之后的内容.mp4
    """
    if not os.path.exists(root_dir):
        print("❌ 错误：根目录不存在，请检查路径！")
        return

    print("=" * 55)
    print(f"📂 开始遍历目录，重命名MP4：{root_dir}")
    print("=" * 55)

    for dirpath, _, filenames in os.walk(root_dir):
        dir_name = os.path.basename(dirpath)

        for filename in filenames:
            # 仅处理 mp4 文件
            if not filename.lower().endswith(".mp4"):
                continue

            if "_" in filename:
                _, file_suffix = filename.split("_", 1)
                new_filename = f"{dir_name}_{file_suffix}"
            else:
                new_filename = f"{dir_name}_{filename}"

            old_path = os.path.join(dirpath, filename)
            new_path = os.path.join(dirpath, new_filename)

            # 防止重名覆盖
            if os.path.exists(new_path):
                print(f"⚠️  已存在，跳过：{new_filename}")
                continue

            # 执行重命名
            try:
                os.rename(old_path, new_path)
                print(f"✅ 重命名成功：{filename} → {new_filename}")
            except Exception as e:
                print(f"❌ 重命名失败 {filename}：{str(e)}")

    print("\n" + "=" * 55)
    print("🎉 所有MP4文件重命名完成！")
    print("=" * 55 + "\n")


def main_menu():
    """功能选择主菜单"""
    # while True:
    # print("========== 批量处理工具 ==========")
    # print("【1】删除指定后缀文件 (jpg/png/txt/gif等)")
    # print("【2】批量重命名目录下 MP4 文件")
    # print("【3】先删除杂文件 → 再重命名 MP4 (连贯执行)")
    # print("【0】退出程序")
    # print("=================================")
    #
    # choice = input("请输入功能编号(0/1/2/3)：").strip()
    # print()
    #
    # if choice == "0":
    #     print("👋 程序已退出")
    #     break
    # elif choice == "1":
    #     batch_delete_files(TARGET_ROOT, DELETE_SUFFIX_LIST)
    # elif choice == "2":
    #     rename_mp4_in_subdirs(TARGET_ROOT)
    # elif choice == "3":
    print(">>> 第一步：删除指定格式文件")
    batch_delete_files(TARGET_ROOT, DELETE_SUFFIX_LIST)
    print(">>> 第二步：重命名所有 MP4 文件")
    rename_mp4_in_subdirs(TARGET_ROOT)
    # else:
    #     print("❌ 输入无效，请选择 0~3 之间的数字\n")


if __name__ == "__main__":
    main_menu()