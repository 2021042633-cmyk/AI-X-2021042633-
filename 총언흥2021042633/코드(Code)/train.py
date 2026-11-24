"""컵 데이터용 YOLOv8n 학습 예시"""
import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="../데이터(Dataset)/data.yaml")
    p.add_argument("--weights", default="yolov8n.pt", help="objects4를 재현할 때는 objects3/best.pt 경로 사용")
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=640, batch=2, device="cpu")

if __name__ == "__main__":
    main()
