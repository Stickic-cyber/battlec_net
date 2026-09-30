#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pure-Python Cap'n Proto wire reader for UNSW Battlecode 2026 "Snake" .replay files.

.replay files are Cap'n Proto messages stored with standard "packed" compression.
This implements just enough wire format to read the `Replay` root, its event list,
and every event struct (field offsets & struct sizes taken from the official
replay-viewer JS codegen in unswbc's replay-viewer.vsix).

Reference: extension/dist/webview/webview.js  (functions hd/fd/Y/pl, struct classes)
"""
from enum import IntEnum

# ---------------- packed compression (port of JS hd / fd) ----------------

def _popcount(x):
    return bin(x).count("1")


def packed_unpack_len(data: bytes) -> int:
    n = 0
    r = 119
    i = 0
    L = len(data)
    while i < L:
        b = data[i]
        if r == 0:
            n += b
            i += 1
            r = 119
        elif r == 255:
            n += b
            i += b * 8 + 1
            r = 119
        else:
            n += 1
            i += _popcount(b) + 1
            r = b
    return n * 8


def packed_unpack(data: bytes) -> bytes:
    out = bytearray(packed_unpack_len(data))
    r = 119
    i = 0
    o = 0
    L = len(data)
    while i < L:
        b = data[i]
        if r == 0:
            o += b * 8
            i += 1
            r = 119
        elif r == 255:
            l = b * 8
            out[o:o + l] = data[i + 1:i + 1 + l]
            o += l
            i += 1 + l
            r = 119
        else:
            i += 1
            bit = 1
            while bit <= 128:
                if b & bit:
                    out[o] = data[i]
                    i += 1
                o += 1
                bit <<= 1
            r = b
    return bytes(out)


# ---------------- primitive little-endian readers ----------------
def _u8(b, o):      return b[o]
def _u16(b, o):     return int.from_bytes(b[o:o+2], "little")
def _u32(b, o):     return int.from_bytes(b[o:o+4], "little")
def _u64(b, o):     return int.from_bytes(b[o:o+8], "little")
def _i32(b, o):     return int.from_bytes(b[o:o+4], "little", signed=True)


# struct data/pointer sizes (bytes, ptrcount) -- from JS Zc(...) in webview.js
SZ = {
    "Replay":              (16, 5),
    "Event":               (8, 1),
    "PlayerAction":        (8, 1),
    "Point":               (8, 0),
    "DebugDraw":           (8, 2),
    "EventRoundStart":     (8, 0),
    "EventTurnStart":      (8, 0),
    "InstructionUsage":    (16, 0),
    "EventPearlCountdown": (8, 1),
    "EventTileChange":     (8, 1),
    "EventDragonAction":   (8, 2),
    "EventEngineLog":      (8, 1),
    "EventDragonLog":      (8, 1),
    "EventDragonIndicator":(8, 1),
    "EventDebugDraw":      (8, 1),
    "EventDragonUpdate":   (8, 2),
    "EventDragonSplit":    (16, 2),
    "EventDragonDeath":    (8, 0),
    "EventSonarPing":      (32, 2),
    "TeamStanding":        (16, 0),
    "GameResult":          (8, 2),
    "Seed":                (16, 0),
}

STRUCT, LIST, FAR, OTHER = 0, 1, 2, 3


class Message:
    def __init__(self, data: bytes):
        unpacked = packed_unpack(data)
        self.raw = unpacked
        n = _u32(unpacked, 0) + 1
        sizes = [_u32(unpacked, 4 + i * 4) * 8 for i in range(n)]
        hdr = (4 + n * 4 + 7) & ~7
        self.segments = []
        pos = hdr
        for s in sizes:
            self.segments.append(unpacked[pos:pos + s])
            pos += s

    def follow(self, seg_id, off):
        """Resolve pointer word at (seg_id,off) -> content (seg_id, off)."""
        seg = self.segments[seg_id]
        word = _u32(seg, off)
        typ = word & 3
        if typ == FAR:
            seg2 = _u32(seg, off + 4)
            poff = (word >> 3) * 8
            if word & 4:
                poff += 8
            return self.follow(seg2, poff)
        content = off + 8 + (_i32(seg, off) >> 2) * 8
        return (seg_id, content)


class Struct:
    def __init__(self, msg, seg_id, data_off, name):
        self.msg = msg
        self.seg = seg_id
        self.data = data_off
        self.nbytes, self.nptr = SZ[name]

    def u8(self, off):   return _u8(self.msg.segments[self.seg], self.data + off)
    def u16(self, off):  return _u16(self.msg.segments[self.seg], self.data + off)
    def i32(self, off):  return _i32(self.msg.segments[self.seg], self.data + off)
    def u32(self, off):  return _u32(self.msg.segments[self.seg], self.data + off)
    def u64(self, off):  return _u64(self.msg.segments[self.seg], self.data + off)
    def bit(self, off):
        return bool(self.msg.segments[self.seg][self.data + (off >> 3)] & (1 << (off & 7)))

    def _ptroff(self, idx):
        ps = (self.nbytes + 7) & ~7
        return self.data + ps + idx * 8

    def ptr(self, idx):
        return self.msg.follow(self.seg, self._ptroff(idx))

    def null_ptr(self, idx):
        return _u32(self.msg.segments[self.seg], self._ptroff(idx)) == 0

    def text(self, idx):
        poff = self._ptroff(idx)
        ctg, coff = self.msg.follow(self.seg, poff)
        seg = self.msg.segments[ctg]
        length = _u32(seg, poff + 4) >> 3
        raw = bytes(seg[coff:coff + length])
        if raw.endswith(b"\x00"):
            raw = raw[:-1]
        return raw.decode("utf-8", errors="replace")


def pointer_list(msg, seg_id, poff):
    """Resolve a list pointer. Returns dict with es, count, content_off,
    and for composite lists the tag elements (data_words, ptr_count).
    The size/count live in the pointer word at poff+4; content offset comes
    from the offset field."""
    seg = msg.segments[seg_id]
    word = _u32(seg, poff)
    if (word & 3) == FAR:
        seg2 = _u32(seg, poff + 4)
        inner = (word >> 3) * 8
        if word & 4:
            inner += 8
        return pointer_list(msg, seg2, inner)
    sizecount = _u32(seg, poff + 4)
    es = sizecount & 7
    count = sizecount >> 3
    content = poff + 8 + (_i32(seg, poff) >> 2) * 8
    info = {"es": es, "count": count, "content": content, "seg": seg_id}
    if es == 7:   # COMPOSITE: tag word sits at content; elements after +8
        info["data_words"] = _u16(seg, content + 4)
        info["ptr_count"] = _u16(seg, content + 6)
        info["first"] = content + 8
        info["count"] = _u32(seg, content) >> 2    # authoritative count in tag
    return info


def struct_list(msg, seg_id, poff):
    """For a composite list, return (count, data_words, ptr_count, first_off)."""
    info = pointer_list(msg, seg_id, poff)
    return info["count"], info["data_words"], info["ptr_count"], info["first"]
