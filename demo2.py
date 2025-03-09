import cv2  # OpenCV库用于图像处理
import imutils  # 图像处理函数库
import numpy as np  # 数值运算库
import pytesseract  # 光学字符识别（OCR）库
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Tesseract可执行文件的路径
from paddleocr import PaddleOCR  # 导入PaddleOCR类

def process_image(img):
    img = cv2.resize(img, (600, 400))  # 将图像调整为600x400像素

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图像
    gray = cv2.bilateralFilter(gray, 13, 15, 15)  # 应用双边滤波器以减少噪声

    edged = cv2.Canny(gray, 30, 200)  # 使用Canny边缘检测器检测图像中的边缘

    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 在边缘图像中查找轮廓
    contours = imutils.grab_contours(contours)  # 获取轮廓
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]  # 按面积排序轮廓并取前10个最大的
    screenCnt = None  # 初始化screenCnt

    for c in contours:  # 遍历轮廓
        peri = cv2.arcLength(c, True)  # 计算轮廓的周长
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)  # 调整近似轮廓的精度

        if len(approx) == 4:  # 如果近似轮廓有4个点
            (x, y, w, h) = cv2.boundingRect(approx)  # 获取近似轮廓的边界框
            aspect_ratio = w / float(h)  # 计算宽高比
            area = cv2.contourArea(approx)  # 计算近似轮廓的面积
            print(f"Contour found with aspect_ratio: {aspect_ratio}, area: {area}")  # 打印宽高比和面积
            if 1.5 < aspect_ratio < 5 and 1000 < area < 300000:  # 约束宽高比和面积
                screenCnt = approx  # 将screenCnt设置为近似轮廓
                break  # 退出循环

    if screenCnt is None:  # 如果没有找到符合条件的轮廓
        detected = 0  # 设置detected为0
        print("No contour detected")  # 打印消息
    else:
        detected = 1  # 设置detected为1

    if detected == 1:  # 如果找到符合条件的轮廓
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)  # 在图像上绘制轮廓

    mask = np.zeros(gray.shape, np.uint8)  # 创建一个与灰度图像形状相同的掩码
    if screenCnt is not None:
        new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1)  # 在掩码上绘制轮廓
        new_image = cv2.bitwise_and(img, img, mask=mask)  # 对图像和掩码进行按位与操作
    else:
        new_image = img  # 如果没有找到符合条件的轮廓，使用原始图像

    (x, y) = np.where(mask == 255)  # 找到掩码中白色像素的坐标
    (topx, topy) = (np.min(x), np.min(y))  # 找到边界框的左上角
    (bottomx, bottomy) = (np.max(x), np.max(y))  # 找到边界框的右下角
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]  # 将灰度图像裁剪到边界框

    text = pytesseract.image_to_string(Cropped, config='--psm 11')  # 对裁剪后的图像进行OCR
    print("programming_fever's License Plate Recognition\n")  # 打印消息
    print("Detected license plate Number is:", text)  # 打印检测到的车牌号码

    """OCR识别"""
    if Cropped is not None:
        # 使用CPU预加载，不用GPU
        ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, ocr_version='PP-OCRv3')
        text = ocr.ocr(Cropped, cls=True)
        for t in text:
            print(t[0][1])
    else:
        print("No valid cropped image for OCR.")

    return img, Cropped

def main():
    choice = input("Enter 'file' to read from file or 'camera' to use webcam: ").strip().lower()

    if choice == 'file':
        img = cv2.imread('car4.jpg', cv2.IMREAD_COLOR)  # 从文件读取图像
        img, Cropped = process_image(img)
        img = cv2.resize(img, (500, 300))  # 调整原始图像的大小
        Cropped = cv2.resize(Cropped, (400, 200))  # 调整裁剪图像的大小
        cv2.imshow('car', img)  # 显示原始图像
        cv2.imshow('Cropped', Cropped)  # 显示裁剪图像
        cv2.waitKey(0)  # 等待按键
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

    elif choice == 'camera':
        cap = cv2.VideoCapture(0)  # 打开摄像头
        while True:
            ret, frame = cap.read()  # 读取摄像头帧
            if not ret:
                break
            img, Cropped = process_image(frame)
            cv2.imshow('car', img)  # 显示原始图像
            cv2.imshow('Cropped', Cropped)  # 显示裁剪图像
            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下'q'键退出
                break
        cap.release()  # 释放摄像头
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

    else:
        print("Invalid choice. Please enter 'file' or 'camera'.")

if __name__ == "__main__":
    main()