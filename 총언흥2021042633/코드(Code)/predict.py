"""학습된 모델로 이미지·영상·카메라 입력을 확인하는 예시"""
import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="이미지, 영상 또는 카메라 입력")
    args = p.parse_args()
    model = YOLO(args.weights)
    model.predict(source=args.source, imgsz=640, device="cpu", show=True)

if __name__ == "__main__":
    main()
