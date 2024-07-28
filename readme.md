# Gesture-Controlled Theremin 🎶

This project is a gesture-controlled audio synthesizer, similar to a Theremin, that uses computer vision and audio synthesis to control sound parameters through hand movements. It employs OpenCV for video capture, MediaPipe for hand tracking, and PyAudio for real-time audio synthesis.

## Features

- **Hand Detection**: Detects up to two hands and uses their positions to control audio parameters. ✋
- **Audio Control**: Control the pitch and volume of a sine wave in real-time. 🔊
- **Visual Feedback**: Provides visual feedback of the detected hand positions and the corresponding audio parameters being controlled. 👀

## Installation

To run this project, you need to install the required Python libraries. You can install them using the following command:

```bash
pip install numpy opencv-python mediapipe pyaudio
```

## Usage

Run the script using Python:

```bash
python main.py
```

Make sure you have a camera connected to your computer for the hand tracking to work. The program will create a window displaying the video feed with hand landmarks. The pitch (frequency) is controlled by the position of the right hand, while the volume is controlled by the position of the left hand.

## How It Works

- **Video Capture**: Captures video from your camera. 📹
- **Hand Tracking**: Uses MediaPipe to detect hand landmarks in real-time. ✋
- **Audio Synthesis**: Generates a sine wave whose frequency and volume are adjusted based on the hand positions detected by the video feed. 🎵

## Configuration

You can adjust several parameters at the beginning of the script:
- `sample_rate`: The sample rate for the audio playback.
- `buffer_duration`: Duration of the audio buffer, which affects latency and smoothness.
- `min_freq`, `max_freq`: Minimum and maximum frequencies corresponding to the pitch controlled by hand position.
- `min_vol`, `max_vol`: Minimum and maximum volumes.

## Exiting the Program

To exit the program, press the ESC key while the video window is active. ⌨️

## Contributing

Feel free to fork this project and submit pull requests. You can also open issues if you find any bugs or have suggestions for improvements. 🙌

## License

MIT License.
