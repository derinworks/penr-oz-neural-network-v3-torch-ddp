import os
import unittest
from unittest.mock import patch, MagicMock
import torch
import ddp


class TestDDP(unittest.TestCase):

    def test_is_ddp_false(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(ddp.is_ddp())

    def test_is_ddp_true(self):
        with patch.dict(os.environ, {"RANK": "0"}):
            self.assertTrue(ddp.is_ddp())

    def test_ddp_rank_default(self):
        with patch.object(ddp.dist, 'is_available', return_value=True), \
             patch.object(ddp.dist, 'is_initialized', return_value=False):
            self.assertEqual(ddp.ddp_rank(), 0)

    def test_ddp_rank_set(self):
        with patch.object(ddp.dist, 'is_available', return_value=True), \
             patch.object(ddp.dist, 'is_initialized', return_value=True), \
             patch.object(ddp.dist, 'get_rank', return_value=2):
            self.assertEqual(ddp.ddp_rank(), 2)

    def test_ddp_local_rank_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ddp.ddp_local_rank(), 0)

    def test_ddp_local_rank_set(self):
        with patch.dict(os.environ, {"LOCAL_RANK": "3"}):
            self.assertEqual(ddp.ddp_local_rank(), 3)

    def test_ddp_world_size_default(self):
        with patch.object(ddp.dist, 'is_available', return_value=True), \
             patch.object(ddp.dist, 'is_initialized', return_value=False):
            self.assertEqual(ddp.ddp_world_size(), 1)

    def test_ddp_world_size_set(self):
        with patch.object(ddp.dist, 'is_available', return_value=True), \
             patch.object(ddp.dist, 'is_initialized', return_value=True), \
             patch.object(ddp.dist, 'get_world_size', return_value=4):
            self.assertEqual(ddp.ddp_world_size(), 4)

    def test_master_proc_true(self):
        with patch.object(ddp.dist, 'is_available', return_value=True), \
             patch.object(ddp.dist, 'is_initialized', return_value=True), \
             patch.object(ddp.dist, 'get_rank', return_value=0):
            self.assertTrue(ddp.master_proc())

    def test_master_proc_false(self):
        with patch.object(ddp.dist, 'is_available', return_value=True), \
             patch.object(ddp.dist, 'is_initialized', return_value=True), \
             patch.object(ddp.dist, 'get_rank', return_value=1):
            self.assertFalse(ddp.master_proc())

    def test_master_proc_default(self):
        with patch.object(ddp.dist, 'is_available', return_value=True), \
             patch.object(ddp.dist, 'is_initialized', return_value=False):
            self.assertTrue(ddp.master_proc())

    @patch('ddp.cuda.device_count')
    @patch('ddp.elastic_launch')
    def test_launch_single_node_ddp_cuda(self, mock_elastic_launch, mock_device_count):
        mock_device_count.return_value = 2
        mock_worker = MagicMock()

        ddp.launch_single_node_ddp("test_run", "cuda", mock_worker, "arg1", "arg2")

        self.assertTrue(mock_elastic_launch.called)
        call_args = mock_elastic_launch.call_args
        config = call_args[0][0]
        self.assertEqual(config.nproc_per_node, 2)
        self.assertEqual(config.run_id, "test_run")

    @patch('ddp.cpu_count')
    @patch('ddp.elastic_launch')
    def test_launch_single_node_ddp_cpu(self, mock_elastic_launch, mock_cpu_count):
        mock_cpu_count.return_value = 8
        mock_worker = MagicMock()

        ddp.launch_single_node_ddp("test_run", "cpu", mock_worker, "arg1")

        self.assertTrue(mock_elastic_launch.called)
        call_args = mock_elastic_launch.call_args
        config = call_args[0][0]
        self.assertEqual(config.nproc_per_node, 4)  # max(1, 8 // 2)

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"version": 1}')
    @patch('logging.config.dictConfig')
    def test_reconfig_logging(self, mock_dict_config, mock_open):
        ddp.reconfig_logging()

        self.assertTrue(mock_open.called)
        self.assertTrue(mock_dict_config.called)


if __name__ == '__main__':
    unittest.main()
