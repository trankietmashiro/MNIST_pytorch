def webcam_inference(model_path="mnist_cnn.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Draw rectangle for digit input area
        x, y, w, h = 200, 100, 200, 200
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract ROI
        roi = gray[y:y+h, x:x+w]

        # Preprocess
        img = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        img = np.invert(img)  # make background black, digit white
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # normalize
        img_tensor = torch.tensor(img).unsqueeze(0).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()

        # Show prediction
        cv2.putText(frame, f"Prediction: {pred}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

        cv2.imshow("Digit Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
# ======================
# Main Menu
# ======================
if __name__ == "__main__":
    if not os.path.exists("mnist_cnn.pth"):
        train_model(epochs=3)
    webcam_inference()
