# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import re

GROUP_SUFFIX_RE = re.compile(r"@[A-Za-z0-9_.-]+$")


def seed_from_nodeid(nodeid: str) -> int:
    canonical_nodeid = GROUP_SUFFIX_RE.sub("", nodeid)
    digest = hashlib.sha256(canonical_nodeid.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)
