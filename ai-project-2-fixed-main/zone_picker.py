import cv2
import numpy as np
import yaml

video_path = r"C:\Users\abhij\Downloads\Telegram Desktop\video_2026-03-21_10-05-06.mp4"

cap = cv2.VideoCapture(video_path)

points = []
all_zones = {}

def mouse_click(event, x, y, flags, param):
    global points, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

        print(f"Point added: {x}, {y}")

        cv2.circle(frame, (x, y), 5, (0,255,0), -1)

        if len(points) > 1:
            cv2.line(frame, points[-2], points[-1], (0,255,0), 2)

        cv2.imshow("Video", frame)

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_click)

print("\n===== ZONE PICKER =====")
print("SPACE  = Next frame (step through)")
print("P      = Play/Pause continuous playback")
print("Click  = Add zone points")
print("C      = Close polygon & name the zone")
print("S      = Save all zones to zones.yaml")
print("ESC    = Exit")
print("========================\n")

paused = True
ret, frame = cap.read()
if ret:
    cv2.imshow("Video", frame)
    cv2.waitKey(1)

while True:

    key = cv2.waitKey(30) & 0xFF

    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)

    if key == 32:  # SPACE = step one frame
        ret, frame = cap.read()
        if not ret:
            break
        paused = True
        cv2.imshow("Video", frame)

    if key == ord('p'):  # P = toggle play/pause
        paused = not paused
        print("PAUSED" if paused else "PLAYING")

    if key == ord('c'):  # close polygon
        if len(points) > 2:
            cv2.line(frame, points[-1], points[0], (0,255,0), 2)
            cv2.imshow("Video", frame)

            print("\nPolygon coordinates:", points)
            zone_name = input("Enter zone name (e.g. left_path, right_path): ").strip()

            if zone_name:
                all_zones[zone_name] = {"points": [list(p) for p in points]}
                print(f"Zone '{zone_name}' saved with {len(points)} points!")
            else:
                print("No name entered, zone discarded.")

            points = []
            print("\nYou can now draw the next zone, or press S to save all.\n")

    if key == ord('s'):  # save all zones to yaml
        if all_zones:
            with open("configs/zones.yaml", "w") as f:
                yaml.dump({"zones": all_zones}, f, default_flow_style=False)
            print(f"\n zones.yaml saved with {len(all_zones)} zones: {list(all_zones.keys())}")
        else:
            print("No zones to save yet.")

    if key == 27:  # ESC = exit
        if all_zones:
            save = input("Save zones before exiting? (y/n): ").strip().lower()
            if save == 'y':
                with open("configs/zones.yaml", "w") as f:
                    yaml.dump({"zones": all_zones}, f, default_flow_style=False)
                print(f"\n zones.yaml saved with {len(all_zones)} zones!")
        break

cap.release()
cv2.destroyAllWindows()