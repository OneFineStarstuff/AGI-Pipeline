import unittest
from io import BytesIO
from fastapi import UploadFile
from main import SpeechProcessor


class TestSpeechProcessor(unittest.TestCase):
    def setUp(self):
        self.speech_processor = SpeechProcessor()

    def test_speech_to_text(self):
        # Use a real (valid) dummy wav file
        with open("test.wav", "rb") as f:
            audio_content = BytesIO(f.read())
        audio_file = UploadFile(filename="test.wav", file=audio_content)
        result = self.speech_processor.speech_to_text(audio_file)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)

    def test_text_to_speech(self):
        text = "Hello world!"
        result = self.speech_processor.text_to_speech(text)
        self.assertIsNone(result)  # Text-to-speech returns None

    def test_text_to_speech_empty_text(self):
        with self.assertRaises(ValueError):
            self.speech_processor.text_to_speech("")


if __name__ == '__main__':
    unittest.main()
