from __future__ import annotations

import argparse
import unittest
from unittest.mock import MagicMock, patch

from discovery2.run_causal_learn import run


class RunCausalLearnCliTests(unittest.TestCase):
    @patch("discovery2.run_causal_learn.ArtifactWriter")
    @patch("discovery2.run_causal_learn.CDNODPipeline")
    @patch("discovery2.run_causal_learn.DatasetPreprocessor")
    def test_run_uses_famafrench_preprocessor(
        self,
        preprocessor_mock: MagicMock,
        pipeline_mock: MagicMock,
        writer_mock: MagicMock,
    ) -> None:
        prepared = MagicMock()
        prepared.run_name = "famafrench"
        prepared.data.shape = (50, 7)
        prepared.context_index.shape = (50, 1)
        preprocessor_mock.prepare_famafrench.return_value = prepared

        artifacts = MagicMock()
        pipeline_instance = pipeline_mock.return_value
        pipeline_instance.run.return_value = artifacts

        args = argparse.Namespace(
            project_root=".",
            dataset="famafrench",
            country="US",
            alpha=0.05,
            max_rows=100,
        )
        run(args)
        preprocessor_mock.prepare_famafrench.assert_called_once()
        pipeline_instance.run.assert_called_once_with(prepared)
        writer_mock.return_value.write_all.assert_called_once_with("famafrench", artifacts)

    @patch("discovery2.run_causal_learn.ArtifactWriter")
    @patch("discovery2.run_causal_learn.CDNODPipeline")
    @patch("discovery2.run_causal_learn.DatasetPreprocessor")
    def test_run_uses_macro_preprocessor(
        self,
        preprocessor_mock: MagicMock,
        pipeline_mock: MagicMock,
        writer_mock: MagicMock,
    ) -> None:
        prepared = MagicMock()
        prepared.run_name = "macro_all"
        prepared.data.shape = (60, 3)
        prepared.context_index.shape = (60, 1)
        preprocessor_mock.prepare_macro.return_value = prepared

        artifacts = MagicMock()
        pipeline_instance = pipeline_mock.return_value
        pipeline_instance.run.return_value = artifacts

        args = argparse.Namespace(
            project_root=".",
            dataset="macro",
            country="all",
            alpha=0.05,
            max_rows=None,
        )
        run(args)
        preprocessor_mock.prepare_macro.assert_called_once()
        pipeline_instance.run.assert_called_once_with(prepared)
        writer_mock.return_value.write_all.assert_called_once_with("macro_all", artifacts)


if __name__ == "__main__":
    unittest.main()
