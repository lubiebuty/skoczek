
import matplotlib.pyplot as plt
import math

def load_video(video_path: str):
    try:
        with open(video_path, "rb") as f:
            first_bytes = f.read(100)
            if not first_bytes:
                print("prześlij w trybie najbardziej zgodnym")
                return
        print("Gites")
    except FileNotFoundError:
        print("Bład ładowania pliku")
        return




def draw_wykres(positions):
    times = [row[0] for row in positions]
    xs = [row[1] for row in positions]
    ys = [row[2] for row in positions]
    offsets = [row[3] for row in positions]

    # Wykres 1: czas vs. pozycja X (piksele)
    plt.figure()
    plt.plot(times, xs, label="X (pix)")
    plt.title("Pozycja X w funkcji czasu")
    plt.xlabel("Czas (s)")
    plt.ylabel("X (pix)")
    plt.legend()

    # Wykres 2: czas vs. pozycja Y (piksele)
    plt.figure()
    plt.plot(times, ys, label="Y (pix)")
    plt.title("Pozycja Y w funkcji czasu")
    plt.xlabel("Czas (s)")
    plt.ylabel("Y (pix)")
    plt.legend()

    # Wykres 3: czas vs. offset (cm)
    plt.figure()
    plt.plot(times, offsets, label="offset (cm)", color="red")
    plt.title("Boczne wychylenie w funkcji czasu")
    plt.xlabel("Czas (s)")
    plt.ylabel("Offset (cm)")
    plt.legend()

    plt.show()

def compute_real_offset_in_cm(px, image_width, horizontal_fov_deg=60.0, distance_cm=120.0):
    """
    Oblicza przybliżoną rzeczywistą odległość (w cm) 'wychylenia' punktu w poziomie
    względem osi optycznej, zakładając:
      - px: tu docelowo lecimy przecięcie kratek
      - image_width: szerokość obrazu w pikselach,
      - horizontal_fov_deg: przybliżony poziomy kąt widzenia kamery (w stopniach),
      - distance_cm: odległość obiektu od kamery wzdłuż osi Z (cm).
    Zwraca wartość przesunięcia w cm (od środka kadru do obiektu).
    """
    # Kąt w stopniach przypadający na 1 piksel
    deg_per_pixel = horizontal_fov_deg / image_width
    # Różnica piksela względem środka kadru
    dx_pixels = px - (image_width / 2.0)
    # Kąt wychylenia (w stopniach)
    angle_deg = dx_pixels * deg_per_pixel
    # Zamiana na radiany
    angle_rad = math.radians(angle_deg)

    # Rzeczywiste przesunięcie w cm przy danej odległości (podstawa z tan(kąta))
    real_offset_cm = distance_cm * math.tan(angle_rad)
    return real_offset_cm
