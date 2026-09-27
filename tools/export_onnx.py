#!/usr/bin/env python3
"""
YOLO 模型导出为 ONNX 格式

作用:
    把 ultralytics 的 .pt 权重转换为 ONNX，供后端以更轻量的方式推理。
    对应作业要求「模型在自己设备上训练完成并转化为 ONNX/NCNN 格式」。

用法:
    python tools/export_onnx.py                     # 导出默认模型 yolo11n
    python tools/export_onnx.py --model yolo11n     # 指定短名称
    python tools/export_onnx.py --model models/best.pt --imgsz 320
    python tools/export_onnx.py --weights models/yolo11n.pt --check   # 导出后自检

导出产物:
    models/yolo11n.onnx

让后端真正使用 ONNX（三选一）:
    1) backend/config.py  设 YOLO_USE_ONNX = True（会优先加载同名 .onnx）
    2) 在代码里 YOLODetector(use_onnx=True)
    3) 直接删除 models/ 下的 .pt，只留 .onnx

说明:
    ONNX 推理在树莓派 5 上通常比 PyTorch 快 20%~40%，CPU 占用更低；
    若需要进一步加速，可继续转 NCNN（见文末提示）。
"""

import argparse
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 YOLO 权重导出为 ONNX 格式",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", default=None,
        help="模型短名称（如 yolo11n）或 .pt 文件路径；默认取 backend/config.py 中的配置",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="导出时的输入尺寸（树莓派可用 320 换取速度）",
    )
    parser.add_argument(
        "--opset", type=int, default=12,
        help="ONNX opset 版本",
    )
    parser.add_argument(
        "--half", action="store_true",
        help="导出 FP16 半精度（体积减半，树莓派 CPU 建议关闭）",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="导出后用后端 YOLODetector 做一次推理自检",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="导出设备；树莓派无 CUDA，默认 cpu（避免 Ultralytics 去抓 GPU 报 Invalid device id）",
    )
    parser.add_argument(
        "--outdir", default=os.path.join(BASE_DIR, "models"),
        help="导出产物目录",
    )
    return parser.parse_args()


def pick_device(requested: str) -> str:
    """优先用用户指定/可用的设备，CUDA 不可用时自动退回 CPU"""
    if requested != "cpu":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return requested
    except Exception:
        pass
    return "cpu"


def resolve_model(model_arg: str) -> str:
    """把用户输入解析成 ultralytics 能直接吃的模型标识"""
    if os.path.isabs(model_arg) or os.path.sep in model_arg:
        if not os.path.exists(model_arg):
            print(f"✗ 未找到模型文件: {model_arg}")
            sys.exit(1)
        return model_arg

    # 纯短名称（如 yolo11n）：先看 models/ 下有没有本地权重
    local_pt = os.path.join(BASE_DIR, "models", f"{model_arg}.pt")
    if os.path.exists(local_pt):
        print(f"  使用本地权重: {local_pt}")
        return local_pt

    print(f"  本地无权重，交由 ultralytics 自动下载: {model_arg}")
    return model_arg


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)

    # 未指定模型时，沿用后端配置，保证导出结果与运行时的模型一致
    if args.model:
        model_ref = resolve_model(args.model)
        model_tag = os.path.splitext(os.path.basename(model_ref))[0]
    else:
        from backend.config import YOLO_MODEL_PATH, YOLO_MODEL_SHORT
        model_ref = resolve_model(YOLO_MODEL_PATH if os.path.exists(YOLO_MODEL_PATH)
                                  else YOLO_MODEL_SHORT)
        model_tag = YOLO_MODEL_SHORT

    print("=" * 62)
    print("YOLO → ONNX 导出")
    print("=" * 62)
    print(f"  模型      : {model_ref}")
    print(f"  输入尺寸  : {args.imgsz}x{args.imgsz}")
    print(f"  opset     : {args.opset}")
    print(f"  半精度    : {'是' if args.half else '否'}")
    print(f"  设备      : {device}")
    print(f"  输出目录  : {args.outdir}")
    print("-" * 62)

    try:
        from ultralytics import YOLO, settings
    except ImportError:
        print("✗ 未安装 ultralytics。请先执行: pip install -r requirements.txt")
        return 1

    os.makedirs(args.outdir, exist_ok=True)

    # 让 ultralytics 把自动下载的权重也放进 models/，而不是散落在项目根目录
    try:
        settings.update({"weights_dir": os.path.abspath(args.outdir)})
    except Exception as exc:
        print(f"  提示: 无法设置 ultralytics 权重目录（{exc}），自动下载的权重可能落在当前目录")

    started = time.time()
    try:
        model = YOLO(model_ref)
        out_path = model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            half=args.half,
            device=device,
        )
    except Exception as exc:
        print(f"✗ 导出失败: {exc}")
        return 1

    elapsed = time.time() - started

    # ultralytics 默认把产物放在权重旁边，统一挪到 models/ 下
    final_path = out_path
    if out_path and os.path.exists(out_path) and os.path.dirname(out_path) != os.path.abspath(args.outdir):
        import shutil
        target = os.path.join(args.outdir, os.path.basename(out_path))
        if os.path.abspath(out_path) != os.path.abspath(target):
            shutil.move(out_path, target)
        final_path = target

    if not final_path or not os.path.exists(final_path):
        print("✗ 导出命令已执行，但未找到产物文件，请检查 ultralytics 输出")
        return 1

    size_mb = os.path.getsize(final_path) / 1e6
    print("-" * 62)
    print(f"✓ 导出成功: {final_path}")
    print(f"  体积    : {size_mb:.2f} MB")
    print(f"  耗时    : {elapsed:.1f} 秒")

    if args.check:
        print("-" * 62)
        print("开始推理自检...")
        try:
            import numpy as np
            from backend.yolo_detector import YOLODetector

            detector = YOLODetector(model_path=final_path, use_onnx=True)
            if not detector.is_loaded:
                # YOLODetector.detect() 内部会吞掉异常并返回空列表，必须显式判失败
                print("✗ ONNX 模型加载失败（详见上方日志）")
                return 1

            dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
            results = detector.detect(dummy)
            print(f"✓ ONNX 模型推理正常（空白测试图检出 {len(results)} 个目标，0 属预期）")
        except ImportError as exc:
            print(f"  跳过自检: 缺少依赖 {exc}")
        except Exception as exc:
            print(f"✗ 自检失败: {exc}")
            return 1

    print("-" * 62)
    print("下一步: 在 backend/config.py 中设置 YOLO_USE_ONNX = True，后端即优先加载 .onnx")
    print("如需更快: 可继续转 NCNN —— yolo export format=ncnn")
    return 0


if __name__ == "__main__":
    sys.exit(main())
