import numpy as np  
import cv2  
import mediapipe as mp  
import pyaudio  
import threading  

# 初始化 MediaPipe  
mp_hands = mp.solutions.hands  
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)  
mp_drawing = mp.solutions.drawing_utils  

# 音频参数  
SAMPLE_RATE = 44100  
BUFFER_DURATION = 0.006  # 每个缓冲区持续时间（秒）  
BUFFER_SAMPLES = int(SAMPLE_RATE * BUFFER_DURATION)  

# 定义音高和音量的范围  
MIN_FREQ, MAX_FREQ = 220, 1880  # A3 to ...  
MIN_VOL, MAX_VOL = 0, 1  

# 初始化 PyAudio  
p = pyaudio.PyAudio()  

# 初始化频率和音量  
current_freq = 440  
current_volume = 0.5  
target_freq = current_freq  
target_volume = current_volume  

# 用于存储前几个频率和音量的值以进行平滑，减少cracks
FREQ_HISTORY = [current_freq] * 5
VOL_HISTORY = [current_volume] * 5  

# 频率和音量变化步长  
FREQ_STEP = 10  # 增大频率步长  
VOL_STEP = 0.05  # 增大音量步长  

# 创建一个锁，用于线程间同步频率和音量的更新  
lock = threading.Lock()  

def smooth_value(history_list, new_value, history_length=5):  
    """  
    使用历史值来平滑新的值  
    """  
    history_list.append(new_value)  
    if len(history_list) > history_length:  
        history_list.pop(0)  
    return np.mean(history_list)  

def get_sine_wave(start_freq, end_freq, start_volume, end_volume, sample_rate, duration):  
    """  
    生成正弦波  
    """  
    t = np.linspace(0, duration, int(sample_rate * duration), False)  
    freq = np.linspace(start_freq, end_freq, int(sample_rate * duration))  
    volume = np.linspace(start_volume, end_volume, int(sample_rate * duration))  
    wave = volume * np.sin(2 * np.pi * freq * t)  
    return wave.astype(np.float32)  

def audio_callback(in_data, frame_count, time_info, status):  
    """  
    音频回调函数  
    """  
    global current_freq, current_volume, target_freq, target_volume, lock, FREQ_HISTORY, VOL_HISTORY  

    with lock:  
        start_freq = current_freq  
        start_volume = current_volume  

        smooth_freq = smooth_value(FREQ_HISTORY, target_freq)  
        smooth_vol = smooth_value(VOL_HISTORY, target_volume)  

        if abs(smooth_freq - current_freq) > FREQ_STEP:  
            next_freq = current_freq + np.sign(smooth_freq - current_freq) * FREQ_STEP  
        else:  
            next_freq = smooth_freq  

        if abs(smooth_vol - current_volume) > VOL_STEP:  
            next_vol = current_volume + np.sign(smooth_vol - current_volume) * VOL_STEP  
        else:  
            next_vol = smooth_vol  

    buffer = get_sine_wave(current_freq, next_freq, current_volume, next_vol, SAMPLE_RATE, frame_count / SAMPLE_RATE)  
    
    with lock:  
        current_freq = next_freq  
        current_volume = next_vol  

    return (buffer.tobytes(), pyaudio.paContinue)  

def main():  
    """  
    主函数  
    """  
    global target_freq, target_volume  
    stream = p.open(format=pyaudio.paFloat32,  
                    channels=1,  
                    rate=SAMPLE_RATE,  
                    frames_per_buffer=BUFFER_SAMPLES,  
                    output=True,  
                    stream_callback=audio_callback)  # 设置音频回调函数

    stream.start_stream()  

    cap = cv2.VideoCapture(0)  
    while cap.isOpened():  
        success, image = cap.read()  
        if not success:  
            continue  

        image = cv2.flip(image, 1)  # 左右镜像翻转  
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        results = hands.process(image_rgb)  

        left_hand_detected, right_hand_detected = False, False  

        if results.multi_hand_landmarks:  
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):  
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)  
                landmarks = hand_landmarks.landmark  

                if idx == 0:  
                    max_x = max([landmark.x for landmark in landmarks])  
                    target_freq = np.interp(max_x, [0, 1], [MIN_FREQ, MAX_FREQ])  
                    left_hand_detected = True  

                    max_x_idx = np.argmax([landmark.x for landmark in landmarks])  
                    max_x_point = landmarks[max_x_idx]  
                    cv2.circle(image, (int(max_x_point.x * image.shape[1]), int(max_x_point.y * image.shape[0])), 10, (0, 255, 0), -1)  
                    cv2.putText(image, f"Frequency: {int(target_freq)} Hz", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  

                elif idx == 1:  
                    max_y = max([landmark.y for landmark in landmarks])  
                    target_volume = np.interp(max_y, [0, 1], [MAX_VOL, MIN_VOL])  
                    right_hand_detected = True  

                    max_y_idx = np.argmax([landmark.y for landmark in landmarks])  
                    max_y_point = landmarks[max_y_idx]  
                    cv2.circle(image, (int(max_y_point.x * image.shape[1]), int(max_y_point.y * image.shape[0])), 10, (255, 0, 0), -1)  
                    cv2.putText(image, f"Volume: {target_volume:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  

            with lock:  
                current_freq = smooth_value(FREQ_HISTORY, target_freq)  
                current_volume = smooth_value(VOL_HISTORY, target_volume)  
                
        else:  
            with lock:  
                target_freq = current_freq  
                target_volume = 0  

        cv2.imshow('MediaPipe Theremin', image)  
        if cv2.waitKey(5) & 0xFF == 27:  
            break  

    cap.release()  
    cv2.destroyAllWindows()  
    stream.stop_stream()  
    stream.close()  
    p.terminate()  

if __name__ == "__main__":  
    main()