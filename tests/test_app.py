
import cv2
import numpy as np

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


# ---------------------------------------
# Test 1: Homepage
# ---------------------------------------

def test_homepage():

    response = client.get("/")

    assert response.status_code == 200
    assert "NeuroScan AI" in response.text


# ---------------------------------------
# Test 2: Health endpoint
# ---------------------------------------

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


# ---------------------------------------
# Test 3: Valid MRI image prediction
# ---------------------------------------

def test_valid_mri_prediction(tmp_path, monkeypatch):

    # Store test logs in a temporary directory
    monkeypatch.setattr(
        "app.LOG_FILE",
        tmp_path / "test_predictions.csv"
    )

    # Create a sample grayscale MRI-like image
    image = np.zeros((128, 128), dtype=np.uint8)

    cv2.circle(
        image,
        (64, 64),
        30,
        150,
        -1
    )

    # Encode image as PNG
    success, encoded_image = cv2.imencode(
        ".png",
        image
    )

    assert success

    response = client.post(
        "/predict",
        files={
            "file": (
                "sample_mri.png",
                encoded_image.tobytes(),
                "image/png"
            )
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [
        "Demented",
        "Non-Demented"
    ]

    assert data["class"] in [0, 1]

    assert data["model"] == "NeuroScan AI"

    # Check whether prediction was logged
    assert (tmp_path / "test_predictions.csv").exists()


# ---------------------------------------
# Test 4: Reject unsupported file type
# ---------------------------------------

def test_invalid_file_extension():

    response = client.post(
        "/predict",
        files={
            "file": (
                "document.txt",
                b"This is not an MRI image",
                "text/plain"
            )
        }
    )

    assert response.status_code == 422


# ---------------------------------------
# Test 5: Reject corrupted image
# ---------------------------------------

def test_corrupted_image():

    response = client.post(
        "/predict",
        files={
            "file": (
                "corrupted.jpg",
                b"This is not a valid image",
                "image/jpeg"
            )
        }
    )

    assert response.status_code == 422
