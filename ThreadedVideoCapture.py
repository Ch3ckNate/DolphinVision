
def video_stream_thread_update():
    global video_stream_thread_frame_available

    while video_stream.isOpened():
        if not is_paused:
            with video_stream_thread_lock:

                success, _ = video_stream.read(buffer_video)

                if not success:
                    video_stream.release()
                    break

                video_stream_thread_frame_available = True

        if not video_is_camera:
            time.sleep(1 / video_stream.get(cv2.CAP_PROP_FPS))
        else:
            time.sleep(0.1)


def video_stream_thread_read():
    with video_stream_thread_lock:
        video_stream_thread_frame_available = False

video_stream_thread = threading.Thread(None, video_stream_thread_update)
video_stream_thread_lock = threading.Lock()
video_stream_thread_frame_available = False
video_stream_thread.start()