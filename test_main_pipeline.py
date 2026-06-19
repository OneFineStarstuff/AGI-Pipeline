import pytest
from main import EnhancedAGIPipeline, TextResponse
from PIL import Image
import io

@pytest.mark.asyncio
async def test_pipeline_nlp():
    pipeline = EnhancedAGIPipeline()
    result = await pipeline.process_nlp("I want to buy some retail items")
    assert "response" in result
    assert "zk_proof" in result
    assert "cae_metadata" in result
    assert result["zk_proof"].status == "VERIFIED"
    assert "Expert_Retail" in result["cae_metadata"].interpretability_summary

@pytest.mark.asyncio
async def test_pipeline_cv():
    pipeline = EnhancedAGIPipeline()
    image = Image.new('RGB', (100, 100), color = 'white')
    result = await pipeline.process_cv(image)
    assert "detections" in result
    assert "cae_metadata" in result

@pytest.mark.asyncio
async def test_pipeline_stt():
    pipeline = EnhancedAGIPipeline()
    from fastapi import UploadFile
    with open("test.wav", "rb") as f:
        audio_file = UploadFile(filename="test.wav", file=io.BytesIO(f.read()))
    result = await pipeline.process_speech_to_text(audio_file)
    assert "response" in result
    assert "cae_metadata" in result
