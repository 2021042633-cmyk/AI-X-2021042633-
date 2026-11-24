"""학습된 모델의 검증 실행 예시"""
import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="../데이터(Dataset)/data.yaml")
    args = p.parse_args()
    model = YOLO(args.weights)
    print(model.val(data=args.data, imgsz=640, device="cpu"))

if __name__ == "__main__":
    main()
