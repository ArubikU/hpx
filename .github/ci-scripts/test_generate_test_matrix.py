#!/usr/bin/env python3
# Copyright (c) 2026 The STE||AR Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Tests of generate_test_matrix.py. Run from anywhere with
#
#   python3 .github/ci-scripts/test_generate_test_matrix.py

import math
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import generate_test_matrix as gtm


def ctest_list(names):
    """What `ctest -N` prints for the given test names."""
    lines = ["Test project /hpx/build"]
    lines += [f"  Test  #{i + 1}: {n}" for i, n in enumerate(names)]
    lines += ["", f"Total Tests: {len(names)}"]
    return "\n".join(lines)


def sample_names():
    """Unit tests of several modules, some of them run for each parcelport."""
    names = []
    for module in ("actions", "algorithms", "collectives", "futures",
            "segmented_algorithms", "serialization"):
        for i in range(30):
            names.append(f"tests.unit.modules.{module}.test_{i:02d}")
    for pp in ("tcp", "mpi", "lci"):
        for i in range(8):
            names.append(
                f"tests.unit.modules.segmented_algorithms.distributed.{pp}."
                f"partitioned_vector_{i}")
    names.append("tests.performance.local.future_overhead_report_perftest")
    return names


def generate(names, buckets, **kw):
    # no exclude files: the default file is not part of this test
    with tempfile.NamedTemporaryFile("w", suffix=".targets",
            delete=False) as f:
        f.write("# nothing excluded\n")
        empty = f.name
    try:
        return gtm.generate_matrix(ctest_list(names), buckets, [empty], **kw)
    finally:
        os.unlink(empty)


def tests_of(entry):
    """The test names a matrix entry selects, read back from its regex."""
    return set(re.sub(r"\\(.)", r"\1", t)
        for t in entry["tests"][2:-2].split("|"))


class ContiguousSplit(unittest.TestCase):
    # The split the workflows use today. It must not change by accident.

    def test_is_consecutive_chunks_of_the_sorted_names(self):
        names = [n for n in sample_names() if not n.endswith("_perftest")]
        got = [sorted(tests_of(e)) for e in generate(names, 7)["include"]]
        ordered = sorted(names)
        size = math.ceil(len(ordered) / 7)
        want = [ordered[i:i + size] for i in range(0, len(ordered), size)]
        self.assertEqual(got, want)

    def test_is_the_default(self):
        names = sample_names()
        self.assertEqual(generate(names, 5),
            generate(names, 5, split="contiguous"))

    def test_entries_are_named_by_first_and_last_module(self):
        entry = generate(sample_names(), 4)["include"][0]
        self.assertRegex(entry["name"], r"^01-\w+-\w+$")
        self.assertEqual(set(entry),
            {"id", "name", "tests", "targets", "count"})


class InterleavedSplit(unittest.TestCase):

    def setUp(self):
        self.names = [n for n in sample_names() if not n.endswith("_perftest")]
        self.matrix = generate(self.names, 6, split="interleaved")["include"]

    def test_every_test_is_in_exactly_one_entry(self):
        seen = [t for e in self.matrix for t in sorted(tests_of(e))]
        self.assertEqual(sorted(seen), sorted(self.names))

    def test_a_target_is_built_in_one_entry_only(self):
        owners = {}
        for e in self.matrix:
            for target in e["targets"].split():
                owners.setdefault(target, set()).add(e["id"])
        self.assertTrue(all(len(v) == 1 for v in owners.values()), owners)

    def test_parcelport_variants_of_a_target_stay_together(self):
        for e in self.matrix:
            tests = tests_of(e)
            for t in tests:
                if ".distributed." in t:
                    variants = {gtm.target_of(x) for x in tests}
                    self.assertIn(gtm.target_of(t), variants)
        # all three variants of one program are in the same entry
        entry = [e for e in self.matrix if any(
            t.endswith("partitioned_vector_3") for t in tests_of(e))]
        self.assertEqual(len(entry), 1)
        self.assertEqual(
            sum(t.endswith("partitioned_vector_3") for t in tests_of(entry[0])),
            3)

    def test_entries_have_nearly_the_same_size(self):
        counts = [e["count"] for e in self.matrix]
        self.assertLessEqual(max(counts) - min(counts), 3)  # a target has 3

    def test_modules_are_spread_over_all_entries(self):
        # in consecutive chunks the 30 tests of a module fall in one or two
        for module in ("segmented_algorithms", "serialization"):
            holding = [e for e in self.matrix
                if any(f".{module}." in t for t in tests_of(e))]
            self.assertEqual(len(holding), len(self.matrix), module)

    def test_names_are_unique_and_numbered(self):
        names = [e["name"] for e in self.matrix]
        self.assertEqual(names, [f"{i + 1:02d}" for i in range(len(names))])

    def test_regex_selects_exactly_the_tests_of_the_entry(self):
        for e in self.matrix:
            rx = re.compile(e["tests"])
            selected = {n for n in self.names if rx.search(n)}
            self.assertEqual(selected, tests_of(e))

    def test_is_deterministic(self):
        again = generate(list(reversed(self.names)), 6, split="interleaved")
        self.assertEqual(again["include"], self.matrix)

    def test_more_entries_than_targets_gives_one_entry_per_target(self):
        names = ["tests.unit.modules.a.x", "tests.unit.modules.a.y"]
        matrix = generate(names, 10, split="interleaved")["include"]
        self.assertEqual([e["count"] for e in matrix], [1, 1])


class Common(unittest.TestCase):

    def test_perftests_and_excluded_names_are_left_out_by_both_splits(self):
        names = sample_names()
        with tempfile.NamedTemporaryFile("w", suffix=".targets",
                delete=False) as f:
            f.write("tests.unit.modules.actions.test_00\n")
            excluded = f.name
        try:
            for split in gtm.SPLIT_STRATEGIES:
                matrix = gtm.generate_matrix(ctest_list(names), 4, [excluded],
                    split=split)["include"]
                every = set().union(*(tests_of(e) for e in matrix))
                self.assertNotIn("tests.unit.modules.actions.test_00", every)
                self.assertFalse([t for t in every if t.endswith("_perftest")])
        finally:
            os.unlink(excluded)

    def test_no_tests_gives_an_empty_matrix(self):
        for split in gtm.SPLIT_STRATEGIES:
            self.assertEqual(generate([], 4, split=split), {"include": []})

    def test_unknown_split_is_refused(self):
        with self.assertRaises(ValueError):
            generate(sample_names(), 4, split="random")


if __name__ == "__main__":
    unittest.main(verbosity=2)
