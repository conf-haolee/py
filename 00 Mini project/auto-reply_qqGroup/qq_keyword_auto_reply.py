import pyautogui
import cv2
import numpy as np
from PIL import ImageGrab
import pytesseract
import time
import sys

def check_text_on_screen(target_text, confidence=0.7, region=None, binarize = False):
    """
    检查屏幕上是否包含目标文字，并计算 OCR 的运行耗时。
    :param target_text: 要检测的目标文字
    :param confidence: 置信度（OCR 结果中的目标文字匹配程度）
    :return: 是否找到目标文字
    """
    start_time = time.time()

    # 截取指定区域的屏幕截图（如果指定了区域）
    screenshot = pyautogui.screenshot(region=region)
    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    screenshot_path = f"screenshot_{int(time.time())}.png"
    # if True:
    #     cv2.imwrite(screenshot_path, screenshot)
    #     print(f"屏幕截图已保存到: {screenshot_path}")

    # 转为灰度图像（OCR 更适合处理灰度图）
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(screenshot_path, screenshot_gray)

    # 如果需要，进行二值化处理
    if binarize:
        _, screenshot_bin = cv2.threshold(screenshot_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        screenshot_gray = screenshot_bin
        print("已对图像进行二值化处理。")
        # 保存二值化图像
        # cv2.imwrite(screenshot_path, screenshot_gray)

    # 如果是 Windows，设置 tesseract.exe 的完整路径
    pytesseract.pytesseract.tesseract_cmd = r'D:\Users\haoLee\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

    # 使用 Tesseract 进行 OCR
    # extracted_text = pytesseract.image_to_string(screenshot_gray, lang='eng')
    extracted_text = pytesseract.image_to_string(screenshot_gray, lang='chi_sim')
    extracted_text = extracted_text.strip()
    print(f"OCR 检测到的文字：\n{extracted_text}")

    elapsed_time = time.time() - start_time
    print(f"OCR 检测耗时: {elapsed_time:.2f} 秒")

    # 检查是否包含目标文字
    return target_text in extracted_text

def find_image_on_screen(target_image_path, confidence=0.8):
    """
    在屏幕上找到与目标图像匹配的位置，并计算图像匹配的运行耗时。
    :param target_image_path: 按钮图片路径
    :param confidence: 图像匹配的信心值（0.0 ~ 1.0）
    :return: 按钮中心点位置 (x, y) 或 None
    """
    start_time = time.time()

    # 截取屏幕截图
    screenshot = pyautogui.screenshot()

    screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # 加载目标图像
    target_image = cv2.imread(target_image_path, cv2.IMREAD_UNCHANGED)
    if target_image is None:
        raise ValueError(f"无法加载图像：{target_image_path}")

    # 如果目标图像有 Alpha 通道，去除之
    if target_image.shape[-1] == 4:  # 包含 Alpha 通道
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGRA2BGR)

    target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

    # 使用模板匹配
    result = cv2.matchTemplate(screenshot_gray, target_image_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    elapsed_time = time.time() - start_time
    print(f"图像匹配耗时: {elapsed_time:.2f} 秒")

    if max_val >= confidence:
        target_height, target_width = target_image_gray.shape[:2]
        target_center = (max_loc[0] + target_width // 2, max_loc[1] + target_height // 2)
        return target_center
    else:
        return None

def click_on_button(position):
    """
    直接按照坐标查找, 点击坐标位置
    """
    pyautogui.moveTo(position)  # 移动鼠标到目标位置
    pyautogui.click()          # 点击鼠标

if __name__ == "__main__":
    # 按钮图片路径
    button_image_path = "sendBtn.png"  # 替换为你的按钮图片路径
    target_text = "让场"              # 要检测的目标文字

    # 提前匹配发送按钮
    position = find_image_on_screen(button_image_path)
    if position:
        print(f"找到按钮，位置: {position}")
        pyautogui.moveTo(position)  # 移动鼠标到目标位置
        pyautogui.click()          # 点击鼠标
    else:
        print("未找到按钮，请调整按钮图片或屏幕显示内容。")
        sys.exit()
    
    print("程序启动，正在实时检测屏幕上的文字...")
    # 检测区域 region = ()
    region = (500, 800, 500, 400) 
    while True:
        try:
            # 检测屏幕上的文字
            if check_text_on_screen(target_text, region = region):
                print(f"检测到文字 '{target_text}'，准备点击按钮...")
                # click_on_button(button_image_path)
                click_on_button(position = position)
                print("按钮点击完成，退出检测。")
                break  # 点击完成后退出循环
            else:
                print(f"未检测到文字 '{target_text}'，继续检测...")

            # 等待 1 秒，防止 CPU 占用过高
            time.sleep(1)
        except KeyboardInterrupt:
            print("程序已被手动终止。")
            break
