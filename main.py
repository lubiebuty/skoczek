import cv2
import time
import csv
import math
import matplotlib.pyplot as plt

from utils import load_video, draw_wykres, compute_real_offset_in_cm


def main():
    video_path = "/Users/bartlomiejostasz/Downloads/IMG_3566.mov"

    # --(1) Sprawdzenie, czy plik wideo jest dostępny w trybie binarnym--
    load_video(video_path)

    # --(2) Inicjalizacja trackera i OpenCV VideoCapture--
    try:
        tracker = cv2.legacy.TrackerCSRT_create()   # tu ogarniam cos zeby była automatyczna detekcja
    except AttributeError:
        tracker = cv2.TrackerCSRT_create()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Nie można otworzyć pliku wideo.")
        return

    # Przygotowanie obiektu do zapisu wideo (np. MP4)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0.0:
        fps = 25.0  # Teraz czy tu jakas pentla do odczytu
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter("tracked_output.mp4", fourcc, fps, (frame_width, frame_height))

    # --(3) Odczyt pierwszej klatki--
    ret, frame = cap.read()
    if not ret:
        print("Nie udało się odczytać pierwszej klatki z wideo.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # Próba automatycznego wykrycia szachownicy
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    template = cv2.imread("checkerboard_template.png", 0)
    template_w, template_h = template.shape[::-1]

    # Match template in the grayscale first frame
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    if max_val >= threshold:
        top_left = max_loc
        bbox = (top_left[0], top_left[1], template_w, template_h)
        print("Automatyczne wykrycie szachownicy (template matching) powiodło się.")
    else:
        print("Nie wykryto szachownicy przez template matching.")
        print("Zaznacz obszar (ROI) do śledzenia ręcznie, a następnie naciśnij ENTER lub SPACJĘ.")
        bbox = cv2.selectROI("Wybierz ROI do śledzenia", frame, False)
        cv2.destroyWindow("Wybierz ROI do śledzenia")

    # Inicjalizacja trackera z obszarem wykrytym (automatycznie) lub zaznaczonym (ręcznie)
    ok = tracker.init(frame, bbox)
    if not ok:
        print("Nie udało się zainicjalizować trackera.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # --(4) Przygotowanie do rejestrowania pozycji--
    positions = []            # lista do przechowywania (t, x, y, offset_cm)
    start_time = time.time()  # czas startu (w sekundach)

    # Parametry do obliczeń odległości w poziomie
    horizontal_fov_deg = 60.0  # zakładany kąt widzenia w poziomie
    distance_z_cm = 120.0      # odległość wzdłuż Z (cm) - stała

    # --(5) Pętla główna do śledzenia obiektu--
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Koniec nagrania

        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            # Rysujemy prostokąt na klatce
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Środek obiektu
            obj_center_x = x + w/2

            # Obliczamy offset w cm
            offset_cm = compute_real_offset_in_cm(
                px=obj_center_x,
                image_width=frame.shape[1],
                horizontal_fov_deg=horizontal_fov_deg,
                distance_cm=distance_z_cm
            )

            cv2.putText(frame, f"Pos: ({x}, {y}) offset: {offset_cm:.1f} cm",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Zapisanie wyników
            current_time = time.time() - start_time
            positions.append((current_time, x, y, offset_cm))
        else:
            cv2.putText(frame, "Obiekt zgubiony!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Zapis klatki do pliku wideo z nałożonymi adnotacjami
        out_video.write(frame)

        cv2.imshow("Śledzenie szachownicy", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # --(6) Zwolnienie zasobów--
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    # --(7) Zapis do pliku CSV--
    with open("positions.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time_s", "x", "y", "offset_cm"])  # nagłówki kolumn
        writer.writerows(positions)

    print("Zapisano dane do pliku positions.csv")

    # --(8) Tworzenie wykresów--
    draw_wykres(positions)

    exit(0)

if __name__ == "__main__":
    main()
