#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import json
import sys

# 配置文件路径
CONFIG_FILE = "./scripts/sync_projects_config.json"

def load_config(config_file):
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"配置文件 {config_file} 不存在，请检查路径。")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"配置文件格式错误: {e}")
        sys.exit(1)

def delete_existing_target(target_path):
    """删除目标路径的现有内容"""
    if os.path.exists(target_path):
        try:
            if os.path.isfile(target_path):
                os.remove(target_path)
            elif os.path.isdir(target_path):
                shutil.rmtree(target_path)
            print(f"已删除目标路径: {target_path}")
        except Exception as e:
            print(f"无法删除目标路径 {target_path}: {e}")
            sys.exit(1)

def sync_directory(source_dir, target_dir):
    """同步单个目录"""
    try:
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        print(f"同步目录 {source_dir} -> {target_dir}")
    except Exception as e:
        print(f"同步目录失败: {source_dir} -> {target_dir}: {e}")

def main():
    # 加载配置
    config = load_config(CONFIG_FILE)
    source_project = config.get("source_project")
    target_projects = config.get("target_projects", [])
    sync_directories = config.get("sync_directories", [])

    if not source_project or not os.path.exists(source_project):
        print(f"源工程路径 {source_project} 不存在，请检查配置。")
        sys.exit(1)

    for target_project in target_projects:
        if not os.path.exists(target_project):
            print(f"目标工程路径 {target_project} 不存在，跳过同步。")
            continue

        print(f"开始同步到目标工程: {target_project}")
        for sync_dir in sync_directories:
            source_dir = os.path.join(source_project, sync_dir)
            target_dir = os.path.join(target_project, sync_dir)

            if not os.path.exists(source_dir):
                print(f"源目录 {source_dir} 不存在，跳过。")
                continue

            # 删除目标目录现有内容
            delete_existing_target(target_dir)

            # 同步目录
            sync_directory(source_dir, target_dir)

    print("同步完成！")

if __name__ == "__main__":
    main()
