import pyautogui
import cv2
import numpy as np
from PIL import ImageGrab
import time

def find_image_on_screen(target_image_path, confidence=0.8):
    """
    在屏幕上找到与目标图像匹配的位置
    :param target_image_path: 按钮图片路径
    :param confidence: 图像匹配的信心值（0.0 ~ 1.0）
    :return: 按钮中心点位置 (x, y) 或 None
    """
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

    if max_val >= confidence:
        target_height, target_width = target_image_gray.shape[:2]
        target_center = (max_loc[0] + target_width // 2, max_loc[1] + target_height // 2)
        return target_center
    else:
        return None

def click_on_button(image_path, confidence=0.8):
    """
    查找屏幕上的按钮并点击
    :param image_path: 按钮图片路径
    :param confidence: 匹配置信度
    """
    # 找到按钮的位置
    position = find_image_on_screen(image_path, confidence)
    if position:
        print(f"找到按钮，位置: {position}")
        pyautogui.moveTo(position)  # 移动鼠标到目标位置
        pyautogui.click()          # 点击鼠标
    else:
        print("未找到按钮，请调整按钮图片或屏幕显示内容。")



if __name__ == "__main__":
    # 按钮图片路径
    button_image_path = "sendBtn2.png"  # 替换为你的按钮图片路径

    # 延迟 3 秒以便切换到目标窗口
    print("程序将在 3 秒后运行...")
    time.sleep(3)

    # 点击按钮
    click_on_button(button_image_path)
