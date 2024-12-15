import cv2
import time
import os

cv2.namedWindow('video', cv2.WINDOW_NORMAL)

# 获取视频设备 也可以从视频文件中读取视频帧率
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 保存视频
save_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
out = cv2.VideoWriter('output_' + save_time + '.mp4', fourcc, 20.0, (640,480))

# 创建保存图片的文件夹
output_dir = "captured_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame_count = 0
while True:
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 0)  # 沿着x轴旋转180
    current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    cv2.putText(frame, current_time, (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (128,128,128), 2, cv2.LINE_AA)

    out.write(frame)

    cv2.imshow('video', frame)
    
    #保存图片
    if frame_count % 10 == 0:
        save_time = time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))
        image_path = os.path.join(output_dir, f"{save_time}_frame_{frame_count}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved {image_path}")
    frame_count += 1

    key = cv2.waitKey(1)
    if(key & 0xFF == ord('q')):
        break

# 释放 VideoCapture
cap.release()
out.release()
cv2.destroyAllWindows()


