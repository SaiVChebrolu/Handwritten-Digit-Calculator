# Handwritten-Digit-Calculator

This project combines computer vision, deep learning, and interactive user input to create a real-time calculator that recognizes handwritten digits from a webcam feed. Using a trained MNIST classifier built with PyTorch in pytorchTest.py, the system captures digits drawn in a designated region on the video stream. The user can then lock in the recognized digit by pressing the space bar, type an arithmetic operator (e.g., `+`, `-`, `*`, `/`), and lock in a second digit. The calculator performs the arithmetic operation on the two operands and displays the result on-screen, which remains visible until the next calculation.

**Key Features:**
- **Real-Time Webcam Integration:** Utilizes OpenCV to capture and process live video, drawing a square region where users can display handwritten digits.
- **Digit Recognition:** Applies a trained MNIST model to recognize digits from the captured region.
- **Interactive Input:** Implements a simple state machine to manage the input sequence: first digit → operator → second digit.
- **Persistent Output Display:** Once the calculation is completed, the result remains on the screen until a new calculation is initiated.
- **Technologies Used:** Python, OpenCV, PyTorch, torchvision, and PIL.
