import unittest
from main import NLPModule


class TestNLPModule(unittest.TestCase):
    def setUp(self):
        self.nlp = NLPModule()

    def test_generate_text_retail(self):
        prompt = "Where can I buy retail stock?"
        result = self.nlp.generate_text(prompt)
        self.assertTrue(hasattr(result, "text"))
        self.assertEqual(result.expert, "Expert_Retail")

    def test_generate_text_finance(self):
        prompt = "What are the best finance investments?"
        result = self.nlp.generate_text(prompt)
        self.assertEqual(result.expert, "Expert_Finance")

    def test_generate_text_general(self):
        prompt = "Hello there!"
        result = self.nlp.generate_text(prompt)
        self.assertEqual(result.expert, "Expert_General")


if __name__ == '__main__':
    unittest.main()
