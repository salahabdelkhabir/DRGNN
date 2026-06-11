import unittest
import os
import sys
import tempfile


class TestDRGNNImport(unittest.TestCase):
    def test_drgnn_import(self):
        from drgnn import DRGNN
        self.assertIsNone(DRGNN)

    def test_version(self):
        from drgnn import __version__
        self.assertEqual(__version__, "1.0.0")

    def test_base_dir_exists(self):
        from drgnn import BASE_DIR
        self.assertTrue(os.path.exists(BASE_DIR))


class TestGraphUtils(unittest.TestCase):
    def test_create_graph(self):
        from drgnn.utils.graph_utils import create_graph
        g = create_graph([(1, 2), (2, 3)])
        self.assertEqual(len(g.nodes), 3)
        self.assertEqual(len(g.edges), 2)

    def test_save_load_graph(self):
        from drgnn.utils.graph_utils import save_graph_to_file, load_graph_from_file, create_graph
        g = create_graph([(1, 2), (2, 3)])
        with tempfile.NamedTemporaryFile(suffix='.edgelist', delete=False) as f:
            tmp_path = f.name
        try:
            save_graph_to_file(g, tmp_path)
            loaded = load_graph_from_file(tmp_path)
            self.assertEqual(len(loaded.nodes), 3)
        finally:
            os.unlink(tmp_path)


class TestConvertStr(unittest.TestCase):
    def test_convert2str(self):
        from drgnn.utils.data_utils import convert2str
        self.assertEqual(convert2str(123), "123.0")
        self.assertEqual(convert2str("abc"), "abc")
        self.assertEqual(convert2str("123_45"), "123_45")


class TestConfig(unittest.TestCase):
    def test_config_paths(self):
        from drgnn.utils.config import BASE_DIR, DATA_DIR, RESULTS_DIR
        self.assertTrue(len(BASE_DIR) > 0)
        self.assertTrue(len(DATA_DIR) > 0)
        self.assertTrue(len(RESULTS_DIR) > 0)

    def test_ensure_dir_exists(self):
        from drgnn.utils.config import ensure_dir_exists
        with tempfile.TemporaryDirectory() as tmp:
            test_dir = os.path.join(tmp, "test_subdir")
            ensure_dir_exists(test_dir)
            self.assertTrue(os.path.exists(test_dir))


if __name__ == "__main__":
    unittest.main()
