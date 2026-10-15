import tempfile
import os
import unittest

from src.review_queue import ReviewQueueManager
from src.storage import DocumentStore


class StorageTests(unittest.TestCase):
    def test_review_items_survive_a_new_manager_on_the_same_db(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.sqlite")
            q1 = ReviewQueueManager(DocumentStore("review_items", path))
            q1.submit_for_review("claim", {"harm_axis": 0.9}, request_id="r1")
            q2 = ReviewQueueManager(DocumentStore("review_items", path))
            self.assertEqual([i.request_id for i in q2.get_pending_reviews()], ["r1"])
            q2.apply_reviewer_decision("r1", "approved")
            q3 = ReviewQueueManager(DocumentStore("review_items", path))
            self.assertEqual(q3.get_pending_reviews(), [])
            self.assertEqual(q3.get_review_item("r1").status, "approved")


if __name__ == "__main__":
    unittest.main()
