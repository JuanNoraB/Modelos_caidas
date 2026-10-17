import cv2
import subprocess


class Procesado:

    def redimensionar(self, imagen, ancho, alto):
        return cv2.resize(imagen, (ancho, alto))

    def reducir_fps(self, fuente, fps_destino):
        # fuente: ruta de video, o camara (0, 1... o url rtsp)
        cap = cv2.VideoCapture(fuente)
        fps_origen = cap.get(cv2.CAP_PROP_FPS) or 30
        paso = max(1, round(fps_origen / fps_destino))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % paso == 0:
                yield frame
            i += 1
        cap.release()

    def reducir_fps_video(self, fuente, fps_destino, salida):
        subprocess.run(["ffmpeg", "-y", "-i", fuente, "-vf", f"fps={fps_destino}",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-an", salida], check=True)
        return salida



