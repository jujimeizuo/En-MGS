#!/bin/bash

# ============================================================================
# 脚本功能: 针对 EndoNeRF 或 StereoMIS 数据集，选择性执行训练、渲染和评估。
# 用法: bash run_pipeline.sh [mode] [dataset]
# 示例 1: bash run_pipeline.sh train cutting        # 训练 EndoNeRF/cutting
# 示例 2: bash run_pipeline.sh render P0            # 渲染 StereoMIS/P0
# 示例 3: bash run_pipeline.sh all pulling          # 运行 pulling 的所有三个模式
# 示例 4: bash run_pipeline.sh all all              # 运行所有数据集的所有模式
# ============================================================================

# --------------------------
# I. 统一配置变量
# --------------------------

# 统一配置文件路径
UNIFIED_CONFIG="arguments/endonerf/default.py"

# 定义预设的输出目录前缀
OUTPUT_BASE="output" 

# --------------------------
# II. 核心函数
# --------------------------

# 函数: 运行训练
run_train() {
    local DATASET_PATH=$1
    local OUTPUT_DIR=$2 # 这是动态生成的输出目录
    local DATASET_NAME=$3 # 这是从路径中提取的唯一标识符

    # 确保输出目录存在
    mkdir -p "$OUTPUT_BASE/$OUTPUT_DIR"

    echo "--- 🚀 任务: 训练 (${DATASET_NAME}) ---"
    
    {
        python train.py -s "$DATASET_PATH" \
            --expname "$OUTPUT_DIR" \
            --configs "$UNIFIED_CONFIG"
    } 2>&1 | tee "$OUTPUT_BASE/$OUTPUT_DIR/log_train.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ 训练 ${DATASET_NAME} 完成。"
    else
        echo "❌ 训练 ${DATASET_NAME} 失败！请查看日志。"
        exit 1
    fi
}

# 函数: 运行渲染
run_render() {
    local OUTPUT_DIR="$OUTPUT_BASE/$1"
    local DATASET_NAME=$2

    echo "--- 🖼️ 任务: 渲染 (${DATASET_NAME}) ---"
    
    {
        python render.py --model_path "$OUTPUT_DIR" \
            --skip_train \
            --configs "$UNIFIED_CONFIG"
    } 2>&1 | tee "$OUTPUT_DIR/log_render.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ 渲染 ${DATASET_NAME} 完成。"
    else
        echo "❌ 渲染 ${DATASET_NAME} 失败！请查看日志。"
        exit 1
    fi
}

# 函数: 运行指标评估
run_metric() {
    local OUTPUT_DIR="$OUTPUT_BASE/$1"
    local METRIC_PARAM=$2
    local DATASET_NAME=$3

    echo "--- 📊 任务: 指标评估 (${DATASET_NAME}, 参数: -p $METRIC_PARAM) ---"
    
    {
        python metrics.py --model_path "$OUTPUT_DIR" -p "$METRIC_PARAM"
    } 2>&1 | tee "$OUTPUT_DIR/log_metric.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✅ 指标评估 ${DATASET_NAME} 完成。"
    else
        echo "❌ 指标评估 ${DATASET_NAME} 失败！请查看日志。"
        exit 1
    fi
}


# --------------------------
# III. 主逻辑
# --------------------------

# 1. 检查参数数量
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "❌ 错误: 参数数量不正确。"
    echo "用法: $0 [train | render | metric | all] [dataset_path] [metric_param (可选)]"
    exit 1
fi

MODE=$(echo "$1" | tr '[:upper:]' '[:lower:]')
DATASET_INPUT_PATH="$2" # 直接使用输入的路径
METRIC_PARAM=${3:-test}

# 2. 动态处理路径和命名

# 定义要去除的前缀
DATASETS_PREFIX="../Datasets/"

# 使用 Bash 字符串替换来移除前缀
# 确保输入的路径 DATASET_INPUT_PATH 始终包含 / 结尾，以便替换正确
# Bash 替换语法: ${variable#pattern} - 从开头移除最短匹配的模式
DS_NAME=${DATASET_INPUT_PATH#$DATASETS_PREFIX}

# 检查 DS_NAME 是否仍有尾部斜杠并移除（例如，如果输入是 "...twice/"）
# Bash 替换语法: ${variable%pattern} - 从结尾移除最短匹配的模式
DS_NAME=${DS_NAME%/}

# 4. 确保expname目录名是小写，以保持一致性
DS_NAME=$(echo "$DS_NAME" | tr '[:upper:]' '[:lower:]')
OUTPUT_DIR="$DS_NAME"

# 3. 运行模式
echo "========================================================"
echo "⚙️ 处理数据集: $DS_NAME"
echo "  -> 路径: $DATASET_INPUT_PATH"
echo "  -> 输出: $OUTPUT_DIR"
echo "========================================================"

case "$MODE" in
    train)
        run_train "$DATASET_INPUT_PATH" "$OUTPUT_DIR" "$DS_NAME"
        ;;
    render)
        run_render "$OUTPUT_DIR" "$DS_NAME"
        ;;
    metric)
        run_metric "$OUTPUT_DIR" "$METRIC_PARAM" "$DS_NAME"
        ;;
    all)
        run_train "$DATASET_INPUT_PATH" "$OUTPUT_DIR" "$DS_NAME"
        run_render "$OUTPUT_DIR" "$DS_NAME"
        run_metric "$OUTPUT_DIR" "$METRIC_PARAM" "$DS_NAME"
        ;;
    *)
        echo "❌ 错误: 无效的模式 '$MODE'。"
        echo "请使用 'train', 'render', 'metric' 或 'all'."
        exit 1
        ;;
esac

echo "========================================================"
echo "🎉 所有指定任务已完成！"
echo "========================================================"