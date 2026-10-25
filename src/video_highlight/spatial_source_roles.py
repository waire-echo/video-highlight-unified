"""Mechanical source-role checks, not a claim about data-use authorization.

The enclosing project supplies a reviewed source SHA256-to-role map separately
from feature records. This map cannot establish its own correctness or approval.
"""
from collections.abc import Mapping
import re


def normalise_source_roles(source_roles):
    if not isinstance(source_roles, Mapping) or not source_roles:
        raise ValueError('A separately reviewed source-role map is required')
    result = {}
    for digest, role in source_roles.items():
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-fA-F]{64}', digest) is None:
            raise ValueError('Source identity must be a SHA256 digest')
        if role not in {'train', 'validation'}:
            raise ValueError('Source-role map contains an unsupported role')
        key = digest.lower()
        if key in result:
            raise ValueError('Duplicate source identity in role map')
        result[key] = role
    return result


def require_source_role(source_sha256, role, source_roles):
    approved_roles = normalise_source_roles(source_roles)
    if not isinstance(source_sha256, str) or re.fullmatch(r'[0-9a-fA-F]{64}', source_sha256) is None:
        raise ValueError('Record source identity must be a SHA256 digest')
    digest = source_sha256.lower()
    if role not in {'train', 'validation'} or approved_roles.get(digest) != role:
        raise ValueError('Source identity is absent from or conflicts with the reviewed role map')
    return digest
