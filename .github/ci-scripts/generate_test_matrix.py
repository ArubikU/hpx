#!/usr/bin/env python3
# Copyright (c) 2026 The STE||AR Group
# Copyright (c) 2026 Surya Teja
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import argparse
import sys
import json
import re
import math
import os

DEFAULT_EXCLUDE_FILE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "workflows", "exclude.targets"
))

def read_list(file_name):
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"exclude file not found: {file_name}")
    entries = []
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(line)
    return entries

def get_excluded_tests(exclude_files):
    excluded = set()
    for exclude_file in exclude_files:
        excluded.update(read_list(exclude_file))
    return excluded

def get_excluded_patterns(exclude_regex_files):
    patterns = []
    for exclude_regex_file in exclude_regex_files:
        patterns.extend(re.compile(p) for p in read_list(exclude_regex_file))
    return patterns

SPLIT_STRATEGIES = ("contiguous", "interleaved")

# distributed tests have .distributed.<parcelport>. in their name which is not
# in the target
def target_of(test_name):
    return re.sub(r'\.distributed\.(tcp|mpi|lci|lcw|gasnet)\.', '.', test_name)

def split_contiguous(tests, num_buckets):
    """Consecutive tests, by name, in buckets of the same size."""
    bucket_size = math.ceil(len(tests) / num_buckets)
    buckets = []
    for i in range(num_buckets):
        start = i * bucket_size
        if start >= len(tests):
            break
        buckets.append(tests[start:(i + 1) * bucket_size])
    return buckets

def split_interleaved(tests, num_buckets):
    """Deal the targets out one by one, so that every bucket gets a mix.

    Tests that belong to the same target (the same program run for different
    parcelports) stay together: the target is built once, in one bucket.
    Tests of neighbouring names usually cost about the same, so a split into
    consecutive tests puts the expensive ones in the same few buckets.
    """
    groups = {}
    for t in tests:
        groups.setdefault(target_of(t), []).append(t)
    buckets = [[] for _ in range(min(num_buckets, len(groups)))]
    for i, target in enumerate(sorted(groups)):
        buckets[i % len(buckets)].extend(groups[target])
    return buckets

def generate_matrix(ctest_output, num_buckets, exclude_files,
        exclude_regex_files=(), split="contiguous"):
    if split not in SPLIT_STRATEGIES:
        raise ValueError(f"unknown split '{split}', use one of "
            f"{', '.join(SPLIT_STRATEGIES)}")
    tests = []
    pattern = re.compile(r"^\s*Test\s*#\d+:\s*(.+)$")

    excluded = get_excluded_tests(exclude_files)
    # matched anywhere in the name, as by ctest --exclude-regex
    excluded_patterns = get_excluded_patterns(exclude_regex_files)

    for line in ctest_output.splitlines():
        match = pattern.match(line)
        if match:
            test_name = match.group(1).strip()
            if test_name in excluded or test_name.endswith('_perftest'):
                continue
            if any(p.search(test_name) for p in excluded_patterns):
                continue
            tests.append(test_name)

    tests.sort()

    if not tests:
        return {"include": []}

    if split == "interleaved":
        buckets = split_interleaved(tests, num_buckets)
    else:
        buckets = split_contiguous(tests, num_buckets)

    matrix = {"include": []}

    for i, bucket_tests in enumerate(buckets):
        test_regex = "^(" + "|".join(re.escape(t) for t in bucket_tests) + ")$"

        bucket_targets = [target_of(t) for t in bucket_tests]

        unique_targets = sorted(list(set(bucket_targets)))
        if len(unique_targets) != 0:
            first_name_parts = unique_targets[0].split('.')
            last_name_parts = unique_targets[-1].split('.')

            first_name_index = -2 if len(first_name_parts) > 1 else -1
            last_name_index = -2 if len(last_name_parts) > 1 else -1

            first_name = first_name_parts[first_name_index]
            last_name = last_name_parts[last_name_index]

            targets_str = " ".join(unique_targets)
            # an interleaved bucket spans all the names, its first and last
            # would say nothing
            name = (f"{i+1:02d}" if split == "interleaved"
                else f"{i+1:02d}-{first_name}-{last_name}")
            matrix["include"].append({
                "id": i + 1,
                "name": name,
                "tests": test_regex,
                "targets": targets_str,
                "count": len(bucket_tests)
            })

    return matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split the tests listed by 'ctest -N' (read from stdin) "
        "into a GitHub Actions job matrix")
    parser.add_argument("num_jobs", nargs="?", type=int, default=10,
        help="number of matrix entries (default: 10)")
    parser.add_argument("--exclude", action="append", default=None,
        metavar="FILE",
        help="file listing tests to leave out, one per line ('#' starts a "
        "comment); may be given more than once (default, if neither "
        "--exclude nor --exclude-regex is given: "
        ".github/workflows/exclude.targets)")
    parser.add_argument("--exclude-regex", action="append", default=[],
        metavar="FILE",
        help="file listing regular expressions, one per line ('#' starts a "
        "comment); tests whose names contain a match are left out, as with "
        "ctest --exclude-regex; may be given more than once")
    parser.add_argument("--split", choices=SPLIT_STRATEGIES,
        default="contiguous",
        help="how the tests are spread over the entries: 'contiguous' "
        "(consecutive tests by name, the default) or 'interleaved' (the "
        "targets dealt out one by one, a mix of cheap and expensive tests in "
        "each entry)")
    args = parser.parse_args()

    try:
        input_data = sys.stdin.read()
        if args.exclude:
            exclude_files = args.exclude
        elif args.exclude_regex:
            exclude_files = []
        else:
            exclude_files = [DEFAULT_EXCLUDE_FILE]

        matrix = generate_matrix(input_data, args.num_jobs, exclude_files,
            args.exclude_regex, args.split)
        print(json.dumps(matrix))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
