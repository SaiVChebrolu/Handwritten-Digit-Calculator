import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Load the saved model
model = torch.load('./my_mnist_model.pt')
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cap = cv2.VideoCapture(0)

box_size = 200
half_box_size = box_size // 2

calc_state = 0
operand1 = None
operator = None
operand2 = None
result = None
last_result_text = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    frame_height, frame_width, _ = frame.shape

    center_x, center_y = frame_width // 2, frame_height // 2
    top_left_x = center_x - half_box_size
    top_left_y = center_y - half_box_size
    bottom_right_x = center_x + half_box_size
    bottom_right_y = center_y + half_box_size

    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

    roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    transformed_image = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        output = model(transformed_image.view(1, 784))
        probabilities = torch.exp(output)
        prediction = torch.argmax(probabilities, dim=1)
        confidence = torch.max(probabilities)

    threshold = 0.7

    if confidence.item() > threshold:
        pred_text = f"Digit: {prediction.item()} ({confidence:.2f})"
    else:
        pred_text = "Low Confidence"

    cv2.putText(frame, pred_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0) if confidence.item() > threshold else (0, 0, 255), 2)

    if calc_state == 0:
        cv2.putText(frame, "Press SPACE to lock FIRST digit", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif calc_state == 1:
        cv2.putText(frame, "Type an operator (+, -, *, /)", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif calc_state == 2:
        cv2.putText(frame, "Press SPACE to lock SECOND digit", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if last_result_text is not None:
        cv2.putText(frame, last_result_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # Waiting for first digit -> space key will confirm the first digit.
    if calc_state == 0 and key == 32:  #space key
        if confidence.item() > threshold:
            operand1 = prediction.item()
            calc_state = 1
            print(f"Operand 1 locked in: {operand1}")
        else:
            print("Low confidence: Unable to lock first digit.")

    elif calc_state == 1 and key in [ord('+'), ord('-'), ord('*'), ord('/')]:
        operator = chr(key)
        calc_state = 2
        print(f"Operator locked in: {operator}")

    elif calc_state == 2 and key == 32:
        if confidence.item() > threshold:
            operand2 = prediction.item()
            if operator == '+':
                result = operand1 + operand2
            elif operator == '-':
                result = operand1 - operand2
            elif operator == '*':
                result = operand1 * operand2
            elif operator == '/':
                result = operand1 / operand2 if operand2 != 0 else "Error: Div 0"

            last_result_text = f"{operand1} {operator} {operand2} = {result}"
            print(last_result_text)

            calc_state = 0
            operand1, operator, operand2, result = None, None, None, None
        else:
            print("Low confidence: Unable to lock second digit.")


    cv2.imshow('Digit Recognition and Calculator', frame)

# close windows
cap.release()
cv2.destroyAllWindows()
