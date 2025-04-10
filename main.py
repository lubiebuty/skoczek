import cv2
import time
import csv
import matplotlib.pyplot as plt

def main():
    video_path = "/Users/bartlomiejostasz/Downloads/IMG_3111.mov"

    # --(1) Sprawdzenie, czy plik wideo jest dostępny w trybie binarnym--
    try:
        with open(video_path, "rb") as f:
            first_bytes = f.read(100)
            if not first_bytes:
                print("The video file seems empty or unreadable.")
                return
        print("Successfully accessed the video file without using external libraries.")
    except FileNotFoundError:
        print("Video file not found. Please check the path.")
        return

    # --(2) Inicjalizacja trackera i OpenCV VideoCapture--
    try:
        tracker = cv2.legacy.TrackerCSRT_create()  # lub cv2.TrackerCSRT_create() w zależności od wersji
    except AttributeError:
        tracker = cv2.TrackerCSRT_create()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Nie można otworzyć pliku wideo.")
        return

    # --(3) Odczyt pierwszej klatki i wybór regionu do śledzenia--
    ret, frame = cap.read()
    if not ret:
        print("Nie udało się odczytać pierwszej klatki z wideo.")
        cap.release()
        cv2.destroyAllWindows()
        return

    bbox = cv2.selectROI("Wybierz obiekt do śledzenia", frame, False)
    cv2.destroyWindow("Wybierz obiekt do śledzenia")

    ok = tracker.init(frame, bbox)
    if not ok:
        print("Nie udało się zainicjalizować trackera.")
        cap.release()
        cv2.destroyAllWindows()
        return

    # --(4) Przygotowanie do rejestrowania pozycji--
    positions = []            # lista do przechowywania (t, x, y)
    start_time = time.time()  # czas startu (w sekundach)

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
            cv2.putText(frame, f"Pozycja: ({x}, {y})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Obliczamy czas od momentu startu
            current_time = time.time() - start_time
            # Zapisujemy do listy
            positions.append((current_time, x, y))
        else:
            cv2.putText(frame, "Obiekt zgubiony!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Śledzenie obiektu", frame)
        # Wyjście po naciśnięciu 'q'
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    # --(6) Zwolnienie zasobów--
    cap.release()
    cv2.destroyAllWindows()

    # --(7) Zapis do pliku CSV--
    with open("positions.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time_s", "x", "y"])  # nagłówki kolumn
        writer.writerows(positions)

    print("Zapisano dane do pliku positions.csv")

    # --(8) Tworzenie wykresów--
    # Przygotowujemy listy czasów, współrzędnych X i współrzędnych Y
    times = [row[0] for row in positions]
    xs = [row[1] for row in positions]
    ys = [row[2] for row in positions]

    # Wykres 1: czas vs. pozycja X
    plt.figure()  # Osobny wykres (nie subplot!)
    plt.plot(times, xs)
    plt.title("Pozycja X w funkcji czasu")
    plt.xlabel("Czas (s)")
    plt.ylabel("X (piksele)")

    # Wykres 2: czas vs. pozycja Y
    plt.figure()  # Osobny wykres (nie subplot!)
    plt.plot(times, ys)
    plt.title("Pozycja Y w funkcji czasu")
    plt.xlabel("Czas (s)")
    plt.ylabel("Y (piksele)")

    # Wyświetlenie obu okien z wykresami
    plt.show()

if __name__ == "__main__":
    main()