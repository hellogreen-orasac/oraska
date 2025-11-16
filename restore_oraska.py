#!/usr/bin/env python3
"""
Oraska v9.2.2 JSON提取脚本
从JSON配置文件生成完整项目结构
"""

import json
import os
import sys
from pathlib import Path

def extract_oraska(json_file: str = "oraska_v922.json"):
    """从JSON提取完整项目"""
    
    print("=" * 70)
    print("Oraska v9.2.2 项目提取器")
    print("=" * 70)
    
    # 1. 读取JSON
    if not os.path.exists(json_file):
        print(f"❌ 错误: 找不到 {json_file}")
        print(f"\n请确保 {json_file} 在当前目录")
        return False
    
    print(f"\n📖 读取配置: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    version = config.get('version', 'unknown')
    print(f"✅ 版本: {version}")
    print(f"✅ 描述: {config.get('description', 'N/A')}")
    
    # 2. 创建项目目录
    print("\n📁 创建目录结构...")
    directories = [
        "oraska",
        "oraska/db",
        "oraska/memory",
        "oraska/rl",
        "oraska/agents",
        "alembic",
        "alembic/versions",
        "checkpoints",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}/")
    
    # 3. 写入所有文件
    print("\n📝 生成文件...")
    files = config.get('files', {})
    file_count = 0
    
    for filepath, content in files.items():
        # 确保父目录存在
        parent_dir = os.path.dirname(filepath)
        if parent_dir:
            Path(parent_dir).mkdir(parents=True, exist_ok=True)
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            # 处理转义的换行符
            content = content.replace('\\n', '\n').replace('\\\\', '\\')
            f.write(content)
        
        file_count += 1
        print(f"  ✓ {filepath}")
    
    print(f"\n✅ 成功生成 {file_count} 个文件")
    
    # 4. 设置可执行权限
    print("\n🔧 设置权限...")
    executable_files = []
    for filepath in files.keys():
        if filepath.endswith('.sh') or filepath.endswith('.py'):
            try:
                os.chmod(filepath, 0o755)
                executable_files.append(filepath)
            except:
                pass
    
    if executable_files:
        print(f"  ✓ 设置 {len(executable_files)} 个文件为可执行")
    
    # 5. 验证关键文件
    print("\n🔍 验证关键文件...")
    critical_files = [
        "requirements.txt",
        "docker-compose.yml",
        "Dockerfile",
        ".env.example",
        "oraska/config.py",
        "oraska/main.py",
        "oraska/orchestrator.py"
    ]
    
    all_present = True
    for filepath in critical_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✓ {filepath} ({size} bytes)")
        else:
            print(f"  ❌ {filepath} 缺失")
            all_present = False
    
    # 6. 生成下一步指令
    print("\n" + "=" * 70)
    print("✅ 项目提取完成!")
    print("=" * 70)
    
    if all_present:
        print("\n📋 下一步操作:")
        print("\n1️⃣  配置环境变量:")
        print("    cp .env.example .env")
        print("    nano .env  # 添加你的 OPENAI_API_KEY")
        print("\n2️⃣  启动服务 (Docker方式):")
        print("    docker-compose up -d")
        print("    # 等待30秒让服务初始化")
        print("    sleep 30")
        print("\n3️⃣  测试系统:")
        print("    curl -X POST http://localhost:8000/tasks/execute \\")
        print("      -H 'Content-Type: application/json' \\")
        print("      -d '{\"description\": \"Design JWT authentication API\"}'")
        print("\n4️⃣  查看指标:")
        print("    curl http://localhost:8000/metrics | jq")
        print("\n📖 详细文档: README.md")
        print("\n⚠️  重要提示:")
        print("   - 确保 Docker 和 Docker Compose 已安装")
        print("   - 确保端口 5432, 6379, 8000 未被占用")
        print("   - 至少需要 4GB 内存和 10GB 磁盘空间")
        
        return True
    else:
        print("\n⚠️  警告: 部分文件缺失，请检查JSON配置")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Oraska v9.2.2 项目提取器')
    parser.add_argument('json_file', nargs='?', default='oraska_v922.json',
                       help='JSON配置文件路径 (默认: oraska_v922.json)')
    parser.add_argument('--verify', action='store_true',
                       help='仅验证JSON格式，不生成文件')
    
    args = parser.parse_args()
    
    if args.verify:
        # 验证模式
        print("🔍 验证JSON格式...")
        try:
            with open(args.json_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"✅ JSON格式正确")
            print(f"✅ 版本: {config.get('version')}")
            print(f"✅ 文件数: {len(config.get('files', {}))}")
            return 0
        except json.JSONDecodeError as e:
            print(f"❌ JSON格式错误: {e}")
            return 1
        except FileNotFoundError:
            print(f"❌ 文件不存在: {args.json_file}")
            return 1
    else:
        # 提取模式
        success = extract_oraska(args.json_file)
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
