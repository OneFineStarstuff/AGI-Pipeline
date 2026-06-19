import unittest
from main import RegulatoryModule, ZKFairnessProof, ContextualAttributionEnvelope

class TestRegulatoryModule(unittest.TestCase):
    def setUp(self):
        self.regulatory = RegulatoryModule()

    def test_verify_zk_fairness(self):
        input_text = "test input"
        result = self.regulatory.verify_zk_fairness(input_text, expert="Expert_Retail")
        self.assertIsInstance(result, ZKFairnessProof)
        self.assertIn(result.status, ["VERIFIED", "FAILED"])
        # Based on new logic: 0.92 + (10 % 5) / 100 = 0.92
        self.assertEqual(result.demographic_parity_score, 0.92)
        self.assertTrue(result.proof_hash.startswith("zkp_"))

    def test_generate_cae_nlp(self):
        result = self.regulatory.generate_cae("NLPModule_Expert_Retail", "test output")
        self.assertIsInstance(result, ContextualAttributionEnvelope)
        # NLPModule_Expert_Retail -> ExpertNode_Retail (split by _ and take last)
        self.assertIn("ExpertNode_Retail", result.contribution_scores)
        self.assertEqual(result.contribution_scores["ExpertNode_Retail"], 0.75)
        self.assertIn("NLPModule_Expert_Retail", result.interpretability_summary)
        self.assertIn("ASA Interpretability Layer", result.interpretability_summary)

    def test_generate_cae_cv(self):
        result = self.regulatory.generate_cae("CVModule", "test detections")
        self.assertIsInstance(result, ContextualAttributionEnvelope)
        self.assertIn("CVModule", result.contribution_scores)
        self.assertEqual(result.contribution_scores["CVModule"], 0.80)
        self.assertIn("CVModule", result.interpretability_summary)

if __name__ == '__main__':
    unittest.main()
