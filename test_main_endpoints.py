import pytest
from fastapi.testclient import TestClient
from main import app, create_access_token
import io
from PIL import Image

client = TestClient(app)


@pytest.fixture
def auth_header():
    token = create_access_token({"sub": "testuser"})
    return {"Authorization": f"Bearer {token}"}


def test_nlp_endpoint(auth_header):
    response = client.post(
        "/process-nlp/",
        json={
            "text": "test"},
        headers=auth_header)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "zk_proof" in data
    assert "cae_metadata" in data


def test_cv_endpoint(auth_header):
    image = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    files = {'file': ('test.png', img_byte_arr, 'image/png')}
    response = client.post(
        "/process-cv-detection/",
        files=files,
        headers=auth_header)
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert "cae_metadata" in data


def test_stt_endpoint(auth_header):
    with open("test.wav", "rb") as f:
        files = {'file': ('test.wav', f, 'audio/wav')}
        response = client.post(
            "/speech-to-text/",
            files=files,
            headers=auth_header)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "cae_metadata" in data
