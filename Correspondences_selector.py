import cv2
import numpy as np
import os

# 이미지 경로
img1_path = '/home/yamal/ImageStitching/a1.jpg'
img2_path = '/home/yamal/ImageStitching/a2.jpg'

# 이미지 로딩
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

if img1 is None or img2 is None:
    raise FileNotFoundError("이미지를 불러올 수 없음")

# 최대 가로 크기 기준 리사이즈 비율
max_display_width = 1000
scale1 = max_display_width / img1.shape[1]
scale2 = max_display_width / img2.shape[1]
scale = min(scale1, scale2)

img1_small = cv2.resize(img1, (int(img1.shape[1] * scale), int(img1.shape[0] * scale)))
img2_small = cv2.resize(img2, (int(img2.shape[1] * scale), int(img2.shape[0] * scale)))

# 결합 이미지 생성
combined = np.hstack((img1_small, img2_small))
combined_display = combined.copy()
w_small = img1_small.shape[1]

# 클릭 관련 변수
pts1, pts2 = [], []
click_count = [0]
done = [False]

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        idx = click_count[0] // 2
        if click_count[0] % 2 == 0 and x < w_small:
            real_x = int(x / scale)
            real_y = int(y / scale)
            pts1.append([real_x, real_y])
            cv2.circle(combined_display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(combined_display, str(idx), (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            click_count[0] += 1
        elif click_count[0] % 2 == 1 and x >= w_small:
            real_x = int((x - w_small) / scale)
            real_y = int(y / scale)
            pts2.append([real_x, real_y])
            cv2.circle(combined_display, (x, y), 5, (255, 0, 0), -1)
            cv2.putText(combined_display, str(idx), (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            click_count[0] += 1

        cv2.imshow(window_name, combined_display)

        if len(pts1) == 15 and len(pts2) == 15:
            print("15쌍 대응점 선택 완료")
            done[0] = True

# 창 생성 및 콜백 연결
window_name = "Select Matching Points"
cv2.namedWindow(window_name)
cv2.imshow(window_name, combined_display)
cv2.setMouseCallback(window_name, click_event)

# 안전한 종료 루프
while not done[0]:
    if cv2.waitKey(1) & 0xFF == 27:  # ESC 누르면 강제 종료
        break

cv2.destroyAllWindows()
cv2.waitKey(1)

# NumPy 배열로 저장
pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

np.save("pts1.npy", pts1)
np.save("pts2.npy", pts2)
print("대응점 저장 완료 → pts1.npy, pts2.npy")
