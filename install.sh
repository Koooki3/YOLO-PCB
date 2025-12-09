#!/bin/bash

# VisualRobot 安装脚本
# 支持多种硬件平台部署

set -e

# 默认配置
DEFAULT_CONFIG="orangepi5_rk3588s"
INSTALL_DIR="/opt/VisualRobot"
CONFIG_FILE="hardware_config.json"

# 显示帮助信息
show_help() {
    echo "VisualRobot 安装脚本"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help             显示帮助信息"
    echo "  -c, --config CONFIG    指定硬件配置名称 (默认: $DEFAULT_CONFIG)"
    echo "  -d, --dir DIR          指定安装目录 (默认: $INSTALL_DIR)"
    echo "  -v, --verbose          显示详细安装过程"
    echo ""
    echo "可用的硬件配置:"
    echo "  orangepi5_rk3588s     OrangePi 5 (RK3588S)"
    echo "  raspberrypi4          Raspberry Pi 4 (BCM2711)"
    echo "  x86_64_pc             x86_64 PC"
    echo ""
}

# 解析命令行参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--config)
                CONFIG_NAME="$2"
                shift 2
                ;;
            -d|--dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift 1
                ;;
            *)
                echo "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 设置默认值
    CONFIG_NAME=${CONFIG_NAME:-$DEFAULT_CONFIG}
    VERBOSE=${VERBOSE:-false}
}

# 检查依赖
check_dependencies() {
    echo "检查依赖..."
    
    # 检查Qt运行时
    if ! command -v qmake &> /dev/null; then
        echo "错误: 未找到 qmake，请确保已安装Qt开发环境"
        exit 1
    fi
    
    # 检查OpenCV
    if ! pkg-config --libs opencv4 &> /dev/null; then
        echo "警告: 未找到OpenCV 4，可能需要手动安装"
    fi
    
    # 检查ONNX Runtime
    if ! ldconfig -p | grep -q libonnxruntime; then
        echo "警告: 未找到ONNX Runtime，可能需要手动安装"
    fi
}

# 安装应用程序
install_app() {
    echo "开始安装 VisualRobot 到 $INSTALL_DIR..."
    
    # 创建安装目录
    sudo mkdir -p $INSTALL_DIR
    sudo mkdir -p $INSTALL_DIR/bin
    sudo mkdir -p $INSTALL_DIR/models
    sudo mkdir -p $INSTALL_DIR/labels
    sudo mkdir -p $INSTALL_DIR/images
    
    # 复制可执行文件
    echo "复制可执行文件..."
    if [ -f "VisualRobot/VisualRobot" ]; then
        sudo cp VisualRobot/VisualRobot $INSTALL_DIR/bin/
    elif [ -f "VisualRobot/VisualRobot.exe" ]; then
        sudo cp VisualRobot/VisualRobot.exe $INSTALL_DIR/bin/
    else
        echo "错误: 未找到可执行文件，请先编译项目"
        exit 1
    fi
    
    # 复制配置文件
    echo "复制配置文件..."
    if [ -f "VisualRobot/$CONFIG_FILE" ]; then
        sudo cp VisualRobot/$CONFIG_FILE $INSTALL_DIR/
    else
        echo "错误: 未找到配置文件 $CONFIG_FILE"
        exit 1
    fi
    
    # 复制模型文件
    echo "复制模型文件..."
    if [ -d "Models" ]; then
        sudo cp -r Models/* $INSTALL_DIR/models/
    else
        echo "警告: 未找到模型目录，跳过模型复制"
    fi
    
    # 复制标签文件
    echo "复制标签文件..."
    if [ -d "Labels" ]; then
        sudo cp -r Labels/* $INSTALL_DIR/labels/
    else
        echo "警告: 未找到标签目录，跳过标签复制"
    fi
    
    # 复制示例图像
    echo "复制示例图像..."
    if [ -d "Img" ]; then
        sudo cp -r Img/* $INSTALL_DIR/images/
    else
        echo "警告: 未找到图像目录，跳过图像复制"
    fi
    
    # 创建启动脚本
    create_start_script
    
    # 设置权限
    sudo chmod +x $INSTALL_DIR/bin/*
    sudo chmod +x $INSTALL_DIR/start.sh
    
    echo "安装完成!"
    echo ""
    echo "使用以下命令启动应用程序:"
    echo "  $INSTALL_DIR/start.sh"
    echo ""
    echo "或使用指定配置启动:"
    echo "  $INSTALL_DIR/start.sh -c <config_name>"
}

# 创建启动脚本
create_start_script() {
    cat > start.sh << EOF
#!/bin/bash

# VisualRobot 启动脚本
# 支持多种硬件平台

set -e

# 默认配置
DEFAULT_CONFIG="$DEFAULT_CONFIG"
INSTALL_DIR="$INSTALL_DIR"
CONFIG_FILE="$CONFIG_FILE"

# 显示帮助信息
show_help() {
    echo "VisualRobot 启动脚本"
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help             显示帮助信息"
    echo "  -c, --config CONFIG    指定硬件配置名称 (默认: $DEFAULT_CONFIG)"
    echo "  -v, --verbose          显示详细启动过程"
    echo ""
    echo "可用的硬件配置:"
    echo "  orangepi5_rk3588s     OrangePi 5 (RK3588S)"
    echo "  raspberrypi4          Raspberry Pi 4 (BCM2711)"
    echo "  x86_64_pc             x86_64 PC"
    echo ""
}

# 解析命令行参数
parse_args() {
    while [[ \$# -gt 0 ]]; do
        case \$1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--config)
                CONFIG_NAME="\$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE=true
                shift 1
                ;;
            *)
                echo "未知选项: \$1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 设置默认值
    CONFIG_NAME=\${CONFIG_NAME:-$DEFAULT_CONFIG}
    VERBOSE=\${VERBOSE:-false}
}

# 主函数
main() {
    parse_args "\$@"
    
    echo "启动 VisualRobot，使用配置: \$CONFIG_NAME"
    
    # 切换到安装目录
    cd \$INSTALL_DIR
    
    # 设置环境变量
    export QT_QPA_PLATFORM=xcb
    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/lib:/opt/MVS/lib/\$(uname -m)
    
    # 启动应用程序
    if [ -f "bin/VisualRobot" ]; then
        bin/VisualRobot
    elif [ -f "bin/VisualRobot.exe" ]; then
        wine bin/VisualRobot.exe
    else
        echo "错误: 未找到可执行文件"
        exit 1
    fi
}

# 执行主函数
main "\$@"
EOF
    
    sudo mv start.sh $INSTALL_DIR/
}

# 主函数
main() {
    parse_args "$@"
    
    if [ "$VERBOSE" = true ]; then
        set -x
    fi
    
    check_dependencies
    install_app
    
    if [ "$VERBOSE" = true ]; then
        set +x
    fi
    
    echo ""
    echo "============================================="
    echo "VisualRobot 安装成功!"
    echo "安装目录: $INSTALL_DIR"
    echo "当前配置: $CONFIG_NAME"
    echo "============================================="
    echo ""
    echo "启动命令: $INSTALL_DIR/start.sh"
    echo ""
}

# 执行主函数
main "$@"