import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

# ========== 修改这里 ==========
# 1. 棋盘格内角点数 (列, 行) —— 屏幕显示的是9x6吗？
pattern_size = (9, 6)

# 2. 用尺子在屏幕上量出的单个方格边长（单位：毫米）
square_size = 30.0   # ⚠️ 改成测量的数字！！！

# 3. 照片路径（不用改）
image_folder = "./*.jpg"
# =================================

print("="*40)
print("开始检测角点...")
print("="*40)

objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []
images = sorted(glob.glob(image_folder))

if len(images) == 0:
    print("❌ 当前文件夹没有找到 .jpg 图片！")
    exit(0)

print(f"✅ 找到 {len(images)} 张图片")

success_count = 0
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"⚠️ 无法读取: {os.path.basename(fname)}，跳过")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        success_count += 1
        objpoints.append(objp)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners_sub)
        print(f"✅ 成功: {os.path.basename(fname)}")
        # 保存角点检测图（报告要用！）
        img_draw = cv2.drawChessboardCorners(img.copy(), pattern_size, corners_sub, ret)
        cv2.imwrite(f"corners_{os.path.basename(fname)}", img_draw)
    else:
        print(f"❌ 失败: {os.path.basename(fname)}")

if success_count < 5:
    print(f"\n❌ 只成功 {success_count} 张，无法标定！")
    print("建议：关灯、屏幕最亮、旋转手机角度重拍几张替换")
    exit(0)

print(f"\n✅ 共成功 {success_count} 张，开始标定...")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n" + "="*40)
print("📷 标定结果（报告抄这里！）")
print("="*40)
print("内参矩阵 K:\n", mtx)
print("\n畸变系数 D (k1, k2, p1, p2, k3):\n", dist.ravel())
print(f"\n重投影误差 (RMS): {ret:.4f} 像素")
print("="*40)

# 去畸变对比图
first_img = cv2.imread(images[0])
h, w = first_img.shape[:2]
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
undistorted = cv2.undistort(first_img, mtx, dist, None, new_camera_mtx)

cv2.imwrite("original.jpg", first_img)
cv2.imwrite("undistorted.jpg", undistorted)

# 生成并排对比图（报告直接贴这张）
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB))
plt.title("原始图像")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
plt.title("去畸变后")
plt.axis('off')
plt.savefig("compare.png", dpi=150, bbox_inches='tight')
print("\n✅ 对比图已保存为 compare.png（报告用）")
print("✅ 角点检测图已保存（corners_开头，挑2张放报告）")