#!/bin/bash

# ============================================================================
# 脚本功能: 针对 EndoNeRF 或 StereoMIS 数据集，选择性执行训练、渲染和评估。
# 用法:
#   - 单数据集: bash run_pipeline.sh [mode] [dataset_path] [metric_param(可选)]
#   - 全部预设: bash run_pipeline.sh all
# 示例 1: bash run_pipeline.sh train ../Datasets/EndoNeRF/cutting_tissues_twice
# 示例 2: bash run_pipeline.sh render ../Datasets/StereoMIS/P3/xxx
# 示例 3: bash run_pipeline.sh all ../Datasets/EndoNeRF/pulling_soft_tissues
# 示例 4: bash run_pipeline.sh all                  # 运行预设列表中的所有数据集
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
            --skip_video \
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

# 函数: 统一处理单个数据集的完整流程
run_single_dataset() {
    local MODE_SINGLE=$1
    local DATASET_PATH=$2
    local METRIC_PARAM_LOCAL=$3

    local DATASETS_PREFIX="../Datasets/"
    local DS_NAME=${DATASET_PATH#$DATASETS_PREFIX}
    DS_NAME=${DS_NAME%/}
    DS_NAME=$(echo "$DS_NAME" | tr '[:upper:]' '[:lower:]')
    local OUTPUT_DIR="$DS_NAME"

    echo "========================================================"
    echo "⚙️ 处理数据集: $DS_NAME"
    echo "  -> 路径: $DATASET_PATH"
    echo "  -> 输出: $OUTPUT_DIR"
    echo "========================================================"

    case "$MODE_SINGLE" in
        train)
            run_train "$DATASET_PATH" "$OUTPUT_DIR" "$DS_NAME"
            ;;
        render)
            run_render "$OUTPUT_DIR" "$DS_NAME"
            ;;
        metric)
            run_metric "$OUTPUT_DIR" "$METRIC_PARAM_LOCAL" "$DS_NAME"
            ;;
        all)
            run_train "$DATASET_PATH" "$OUTPUT_DIR" "$DS_NAME"
            run_render "$OUTPUT_DIR" "$DS_NAME"
            run_metric "$OUTPUT_DIR" "$METRIC_PARAM_LOCAL" "$DS_NAME"
            ;;
        *)
            echo "❌ 错误: 无效的模式 '$MODE_SINGLE'。"
            echo "请使用 'train', 'render', 'metric' 或 'all'."
            exit 1
            ;;
    esac

    echo "========================================================"
    echo "🎉 数据集 ${DS_NAME} 的任务已完成！"
    echo "========================================================"
}


# --------------------------
# III. 主逻辑
# --------------------------

# 1. 批量模式：仅 1 个参数且为 all
if [ "$#" -eq 1 ]; then
    MODE=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    if [ "$MODE" != "all" ]; then
        echo "❌ 错误: 参数数量不正确。"
        echo "用法: $0 [train | render | metric | all] [dataset_path] [metric_param (可选)]"
        echo "或:   $0 all   # 运行预设列表中的所有数据集"
        exit 1
    fi

    # 与原 run_all.sh 相同的预设数据集列表
    PRESET_DATASETS=(
        "../Datasets/EndoNeRF/cutting_tissues_twice"
        "../Datasets/EndoNeRF/pulling_soft_tissues"
        "../Datasets/StereoMIS/P2_7/stereo_P2_7_550_936"
        "../Datasets/StereoMIS/P3/stereo_P3_10060_10360"
        "../Datasets/StereoMIS/P3/stereo_P3_13100_13367"
        "../Datasets/StereoMIS/P3/stereo_P3_15100_15468"
        "../Datasets/StereoMIS/P2_6/stereo_P2_6_11634_11801"
        "../Datasets/StereoMIS/P2_6/stereo_P2_6_9750_9900"
    )

    echo "✅ 进入批量模式，预设数据集数量: ${#PRESET_DATASETS[@]}"

    for DATASET_INPUT_PATH in "${PRESET_DATASETS[@]}"; do
        run_single_dataset "$MODE" "$DATASET_INPUT_PATH" "test"
    done

    echo "========================================================"
    echo "🎉 全部预设数据集处理完成！"
    echo "========================================================"
    exit 0
fi

# 2. 常规单数据集模式
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "❌ 错误: 参数数量不正确。"
    echo "用法: $0 [train | render | metric | all] [dataset_path] [metric_param (可选)]"
    echo "或:   $0 all   # 运行预设列表中的所有数据集"
    exit 1
fi

MODE=$(echo "$1" | tr '[:upper:]' '[:lower:]')
DATASET_INPUT_PATH="$2" # 直接使用输入的路径
METRIC_PARAM=${3:-test}

run_single_dataset "$MODE" "$DATASET_INPUT_PATH" "$METRIC_PARAM"

echo "========================================================"
echo "🎉 所有指定任务已完成！"
echo "========================================================"