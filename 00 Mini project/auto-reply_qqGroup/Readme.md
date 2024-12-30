# auto_reply_qqgroup



一种低效的 qq群自动关键词检测 并回复程序

**背景：**

学校的羽毛球场地紧张，不定时在qq群里放出空闲场地，先到先得。本人次次抢不到，一怒之下。

**主要技术原理：**

1. 图像模板匹配，匹配屏幕中的qq 发送按钮
2. OCR 字符检测，识别屏幕中的字符，用于匹配关键词
3. pyautogui 库执行鼠标点击操作

使用方法：



可能遇到的问题：

teeseract-ocr 安装错误：参考 [安装Tesseract-OCR并配置TESSERACT_HOME和TESSDATA_PREFIX实用教程](https://teachcourse.cn/3583.html)

> 引用

