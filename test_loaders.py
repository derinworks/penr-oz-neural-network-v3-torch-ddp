import os
import unittest
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock
from loaders import Downloader, Loader, DATA_FOLDER


class TestDownloader(unittest.TestCase):

    def setUp(self):
        # Create a temporary data folder for tests
        self.test_data_folder = tempfile.mkdtemp()
        self.original_data_folder = DATA_FOLDER

    def tearDown(self):
        # Clean up temporary folder
        if os.path.exists(self.test_data_folder):
            shutil.rmtree(self.test_data_folder)

    @patch('loaders.multiprocessing.Pool')
    @patch('loaders.DATA_FOLDER')
    @patch('loaders.load_dataset')
    def test_download(self, mock_load_dataset, mock_data_folder, mock_pool_class):
        mock_data_folder.__str__ = lambda self: self.test_data_folder
        
        # Mock dataset
        mock_ds = {"text": ["Hello world", "This is a test"]}
        mock_load_dataset.return_value = mock_ds
        
        # Mock the Pool and imap
        mock_pool = MagicMock()
        mock_pool_class.return_value.__enter__.return_value = mock_pool
        mock_pool.imap.return_value = [[1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11]]
        
        # Create downloader (use shard_size >= 100 to avoid division by zero in download())
        downloader = Downloader("test_dataset", shard_size=100, encoding="gpt2")
        
        with patch.object(downloader, '_save') as mock_save:
            downloader.download("path/to/dataset", "default", "train")
            
            # Verify save was called
            self.assertTrue(mock_save.called)

    @patch('loaders.DATA_FOLDER', 'test_data')
    @patch('loaders.np.save')
    def test_downloader_save(self, mock_np_save):
        downloader = Downloader("test_ds", shard_size=100, encoding="gpt2")
        
        # Call _save
        downloader._save(0, [1, 2, 3, 4, 5])
        
        # Verify np.save was called
        self.assertTrue(mock_np_save.called)
        call_args = mock_np_save.call_args[0]
        self.assertIn("test_ds_000000", call_args[0])


class TestLoader(unittest.TestCase):

    def setUp(self):
        # Create a temporary data folder with test shards
        self.test_data_folder = tempfile.mkdtemp()
        self.dataset_id = "test_dataset"
        
        # Create test shards
        self.shard1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint16)
        self.shard2 = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20], dtype=np.uint16)
        
        np.save(os.path.join(self.test_data_folder, f"{self.dataset_id}_000000.npy"), self.shard1)
        np.save(os.path.join(self.test_data_folder, f"{self.dataset_id}_000001.npy"), self.shard2)

    def tearDown(self):
        # Clean up temporary folder
        if os.path.exists(self.test_data_folder):
            shutil.rmtree(self.test_data_folder)

    @patch('loaders.DATA_FOLDER')
    def test_loader_init(self, mock_data_folder):
        mock_data_folder.__str__.return_value = self.test_data_folder
        
        with patch('loaders.os.listdir', return_value=[f"{self.dataset_id}_000000.npy", f"{self.dataset_id}_000001.npy"]):
            loader = Loader(self.dataset_id)
            
            self.assertEqual(len(loader.shards), 2)
            self.assertEqual(loader.shard_idx, 0)
            self.assertEqual(loader.token_idx, 0)

    @patch('loaders.DATA_FOLDER')
    def test_loader_list(self, mock_data_folder):
        mock_data_folder.__str__.return_value = self.test_data_folder
        
        with patch('loaders.os.listdir', return_value=[f"{self.dataset_id}_000000.npy", f"{self.dataset_id}_000001.npy"]):
            loader = Loader(self.dataset_id)
            shards = loader.list()
            
            self.assertEqual(len(shards), 2)
            self.assertIn(f"{self.dataset_id}_000000.npy", shards)

    @patch('loaders.DATA_FOLDER')
    def test_loader_next_batch(self, mock_data_folder):
        mock_data_folder.__str__.return_value = self.test_data_folder
        
        with patch('loaders.os.listdir', return_value=[f"{self.dataset_id}_000000.npy"]):
            with patch('loaders.os.path.join', return_value=os.path.join(self.test_data_folder, f"{self.dataset_id}_000000.npy")):
                loader = Loader(self.dataset_id, buffer_size=4, idx_offset=4)
                
                input_arr, target_arr = loader.next_batch()
                
                self.assertIsNotNone(input_arr)
                self.assertIsNotNone(target_arr)
                self.assertEqual(len(input_arr), 4)
                self.assertEqual(len(target_arr), 4)

    @patch('loaders.DATA_FOLDER')
    def test_loader_delete(self, mock_data_folder):
        mock_data_folder.__str__.return_value = self.test_data_folder
        
        test_shard_path = os.path.join(self.test_data_folder, f"{self.dataset_id}_temp.npy")
        np.save(test_shard_path, np.array([1, 2, 3], dtype=np.uint16))
        
        with patch('loaders.os.listdir', return_value=[f"{self.dataset_id}_temp.npy"]):
            loader = Loader(self.dataset_id)
            
            with patch('loaders.os.remove') as mock_remove:
                loader.delete()
                self.assertTrue(mock_remove.called)

    @patch('loaders.DATA_FOLDER')
    def test_loader_next_batch_shard_wraparound(self, mock_data_folder):
        # Set DATA_FOLDER to test directory directly
        mock_data_folder.__str__.return_value = self.test_data_folder
        
        # Store original join to avoid recursion
        import os.path as path_module
        original_join = path_module.join
        
        # Create loader with 2 shards
        with patch('loaders.os.listdir', return_value=[f"{self.dataset_id}_000000.npy", f"{self.dataset_id}_000001.npy"]):
            with patch('loaders.os.path.join', side_effect=lambda *args: original_join(self.test_data_folder, args[-1])):
                loader = Loader(self.dataset_id, buffer_size=4, idx_offset=8)
                
                # First call loads shard 0
                input_arr, target_arr = loader.next_batch()
                self.assertEqual(loader.shard_idx, 0)
                
                # Second call should advance to shard 1
                input_arr, target_arr = loader.next_batch()
                self.assertEqual(loader.shard_idx, 1)
                
                # Third call should wrap back to shard 0
                input_arr, target_arr = loader.next_batch()
                self.assertEqual(loader.shard_idx, 0)


if __name__ == '__main__':
    unittest.main()
