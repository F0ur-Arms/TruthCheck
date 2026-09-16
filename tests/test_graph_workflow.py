import unittest
from graph.state import TruthCheckState
from graph.nodes import ingest_node, route_and_decompose_node, assess_risk_node
from graph.workflow import create_truthcheck_graph


class LangGraphWorkflowTests(unittest.TestCase):
    def test_ingest_node_cleans_text(self):
        state = TruthCheckState(raw_input="Subah khali pet garam pani fatigue kam karta hai")
        res = ingest_node(state)
        self.assertIn("cleaned_text", res)
        self.assertIn("morning", res["cleaned_text"])

    def test_medical_advice_bypasses_graph(self):
        state = TruthCheckState(raw_input="I have stage 3 CKD, should I stop drinking whey protein?", cleaned_text="i have stage 3 ckd, should i stop drinking whey protein?")
        res = route_and_decompose_node(state)
        self.assertEqual(res["route"], "medical_advice")
        self.assertIsNotNone(res.get("final_response"))
        self.assertEqual(res["final_response"]["verdict"], "NOT_A_FACT_CHECK")

    def test_graph_compilation_and_execution(self):
        app = create_truthcheck_graph()
        initial_state = {"raw_input": "Warm water improves digestion."}
        config = {"configurable": {"thread_id": "test-thread-1"}}
        
        final_output = app.invoke(initial_state, config=config)
        self.assertIn("final_response", final_output)
        self.assertIn(final_output["final_response"]["verdict"], {"SUPPORTED", "MOSTLY_SUPPORTED", "INSUFFICIENT_EVIDENCE"})


if __name__ == "__main__":
    unittest.main()
