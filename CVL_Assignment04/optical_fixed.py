import cv2
import numpy as np


def compute_gradients(frame):
    kernel_x = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]], dtype=np.float32)
    Ix = cv2.filter2D(frame.astype(np.float32), -1, kernel_x)
    Iy = cv2.filter2D(frame.astype(np.float32), -1, kernel_y)
    return Ix, Iy


def lucas_kanade_point(Ix, Iy, It, x, y, window_size=15):
    half = window_size // 2
    h, w = Ix.shape
    x1, x2 = max(0, x - half), min(w, x + half + 1)
    y1, y2 = max(0, y - half), min(h, y + half + 1)
    Ix_win = Ix[y1:y2, x1:x2].flatten()
    Iy_win = Iy[y1:y2, x1:x2].flatten()
    It_win = It[y1:y2, x1:x2].flatten()
    A = np.stack([Ix_win, Iy_win], axis=1)
    b = -It_win
    ATA = A.T @ A
    eigenvalues = np.linalg.eigvals(ATA)
    min_eig = np.min(np.abs(eigenvalues))
    if min_eig < 1e-3:
        return 0.0, 0.0, False
    ATb = A.T @ b
    flow = np.linalg.solve(ATA, ATb)
    return float(flow[0]), float(flow[1]), True


def detect_corners(gray, max_corners=80, quality=0.3, min_dist=10, mask=None):
    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_dist,
        mask=mask
    )
    if pts is None:
        return np.empty((0, 2), dtype=np.int32)
    return pts.reshape(-1, 2).astype(np.int32)


def track_points(old_gray, new_gray, points, window_size=15, max_move=30):
    Ix, Iy = compute_gradients(old_gray)
    It = new_gray.astype(np.float32) - old_gray.astype(np.float32)
    new_points = []
    statuses   = []
    for (x, y) in points:
        u, v, valid = lucas_kanade_point(Ix, Iy, It, x, y, window_size)
        if valid and abs(u) < max_move and abs(v) < max_move:
            new_x = int(x + u)
            new_y = int(y + v)
            h, w = old_gray.shape
            if 0 <= new_x < w and 0 <= new_y < h:
                new_points.append((new_x, new_y))
                statuses.append(True)
                continue
        new_points.append((x, y))
        statuses.append(False)
    return np.array(new_points), np.array(statuses)


def get_bbox(valid_pts, shape, padding=12):
    if len(valid_pts) == 0:
        return None
    xs, ys = valid_pts[:, 0], valid_pts[:, 1]
    return (max(0, int(xs.min()) - padding),
            max(0, int(ys.min()) - padding),
            min(shape[1] - 1, int(xs.max()) + padding),
            min(shape[0] - 1, int(ys.max()) + padding))

def expand_bbox(bbox, shape, roi_w, roi_h):
    """Pastikan bbox tidak lebih kecil dari ukuran ROI awal."""
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    bw = max(x2 - x1, roi_w)
    bh = max(y2 - y1, roi_h)
    nx1 = max(0, cx - bw // 2)
    ny1 = max(0, cy - bh // 2)
    nx2 = min(shape[1] - 1, nx1 + bw)
    ny2 = min(shape[0] - 1, ny1 + bh)
    return nx1, ny1, nx2, ny2

def bbox_to_mask(bbox, shape):
    mask = np.zeros((shape[0], shape[1]), dtype=np.uint8)
    if bbox:
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 255
    return mask

def draw_bbox(frame, bbox):
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = "Tracked Object"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

cap = cv2.VideoCapture("skateboard.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS video: {fps}")

skip_to_second = 3
target_frame   = int(fps * skip_to_second)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

ret, first_frame = cap.read()
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

print("Pilih area objek yang mau di-track, lalu tekan ENTER")
roi = cv2.selectROI("Pilih Area", first_frame, False, False)
cv2.destroyAllWindows()

x, y, w, h = roi
ROI_W, ROI_H = w, h  

roi_mask = np.zeros_like(old_gray)
roi_mask[y:y+h, x:x+w] = 255
points = detect_corners(old_gray, mask=roi_mask)
print(f"Titik awal terdeteksi: {len(points)}")

current_bbox  = (x, y, x + w, y + h)
colors        = np.random.randint(50, 255, (200, 3)).tolist()
trail         = np.zeros_like(first_frame)
frame_idx     = 0
REFRESH_EVERY = 40

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if len(points) > 0:
        new_points, statuses = track_points(old_gray, new_gray, points)

        for i, (old_pt, new_pt, ok) in enumerate(zip(points, new_points, statuses)):
            if not ok:
                continue
            col = colors[i % len(colors)]
            cv2.line(trail, tuple(old_pt), tuple(new_pt), col, 2)
            cv2.circle(frame, tuple(new_pt), 4, col, -1)

        frame = cv2.add(frame, trail)
        valid_pts = new_points[statuses]

        raw_bbox     = get_bbox(valid_pts, frame.shape, padding=12)
        current_bbox = expand_bbox(raw_bbox, frame.shape, ROI_W, ROI_H)
        draw_bbox(frame, current_bbox)

        points = valid_pts

    if len(points) < 10 or frame_idx % REFRESH_EVERY == 0:
        dyn_mask = bbox_to_mask(current_bbox, new_gray.shape)
        points   = detect_corners(new_gray, mask=dyn_mask)
        trail    = np.zeros_like(frame)

    cv2.putText(frame, f"Points: {len(points)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame : {frame_idx}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, "Optical Flow (from scratch)", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)

    cv2.imshow("Optical Flow - From Scratch", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    old_gray = new_gray.copy()

cap.release()
cv2.destroyAllWindows()