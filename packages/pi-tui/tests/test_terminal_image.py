"""Tests for pi.tui.terminal_image."""

from __future__ import annotations

import base64
import struct

import pytest

from pi.tui.terminal_image import (
    CellDimensions,
    ImageDimensions,
    ImageRenderOptions,
    TerminalCapabilities,
    allocate_image_id,
    calculate_image_rows,
    delete_all_kitty_images,
    delete_kitty_image,
    encode_iterm2,
    encode_kitty,
    get_capabilities,
    get_cell_dimensions,
    get_gif_dimensions,
    get_image_dimensions,
    get_jpeg_dimensions,
    get_png_dimensions,
    get_webp_dimensions,
    image_fallback,
    is_image_line,
    reset_capabilities_cache,
    set_cell_dimensions,
)


# ---------------------------------------------------------------------------
# Helpers — minimal binary image headers
# ---------------------------------------------------------------------------


def _make_png_header(width: int, height: int) -> bytes:
    """Build a minimal PNG file header (first 24 bytes) with given dimensions."""
    # PNG signature (8 bytes)
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR chunk: length (4) + "IHDR" (4) + width (4) + height (4) = 16 bytes
    ihdr_length = struct.pack(">I", 13)  # IHDR data is always 13 bytes
    ihdr_type = b"IHDR"
    ihdr_width = struct.pack(">I", width)
    ihdr_height = struct.pack(">I", height)
    # Remaining 5 bytes of IHDR data (bit depth, color type, etc.)
    ihdr_rest = b"\x08\x02\x00\x00\x00"
    return sig + ihdr_length + ihdr_type + ihdr_width + ihdr_height + ihdr_rest


def _make_jpeg_data(width: int, height: int) -> bytes:
    """Build minimal JPEG binary with SOF0 marker encoding dimensions."""
    # SOI marker
    soi = b"\xff\xd8"
    # APP0 marker (minimal, just to have something between SOI and SOF)
    app0_marker = b"\xff\xe0"
    app0_length = struct.pack(">H", 2)  # minimal length
    # SOF0 marker
    sof0_marker = b"\xff\xc0"
    # SOF0 length (minimum: 2 + 1 + 2 + 2 + 1 = 8, but we only need header)
    sof0_length = struct.pack(">H", 8)
    sof0_precision = b"\x08"
    sof0_height = struct.pack(">H", height)
    sof0_width = struct.pack(">H", width)
    sof0_components = b"\x03"
    return (
        soi
        + app0_marker
        + app0_length
        + sof0_marker
        + sof0_length
        + sof0_precision
        + sof0_height
        + sof0_width
        + sof0_components
    )


def _make_gif_data(width: int, height: int) -> bytes:
    """Build minimal GIF89a header."""
    sig = b"GIF89a"
    w = struct.pack("<H", width)
    h = struct.pack("<H", height)
    return sig + w + h


def _make_webp_vp8_data(width: int, height: int) -> bytes:
    """Build minimal WebP (VP8 lossy) binary."""
    # RIFF header
    riff = b"RIFF"
    webp = b"WEBP"
    chunk_type = b"VP8 "
    # VP8 bitstream starts at offset 20; dimensions at offset 26-30
    # We need to fill: RIFF(4) + size(4) + WEBP(4) + VP8 (4) + chunk_size(4)
    # + 6 bytes padding + width(2) + height(2)
    # Total minimum: 30 bytes
    # VP8 chunk payload: 6 bytes of frame tag + width(2) + height(2) = 10
    vp8_payload = b"\x00" * 6 + struct.pack("<H", width) + struct.pack("<H", height)
    chunk_size = struct.pack("<I", len(vp8_payload))
    file_size = struct.pack("<I", 4 + 4 + 4 + len(vp8_payload))  # after RIFF+size
    return riff + file_size + webp + chunk_type + chunk_size + vp8_payload


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ---------------------------------------------------------------------------
# get_png_dimensions
# ---------------------------------------------------------------------------


class TestGetPngDimensions:
    def test_valid_png(self) -> None:
        data = _b64(_make_png_header(320, 240))
        dims = get_png_dimensions(data)
        assert dims is not None
        assert dims.width_px == 320
        assert dims.height_px == 240

    def test_large_dimensions(self) -> None:
        data = _b64(_make_png_header(4096, 2160))
        dims = get_png_dimensions(data)
        assert dims is not None
        assert dims.width_px == 4096
        assert dims.height_px == 2160

    def test_one_by_one_png(self) -> None:
        data = _b64(_make_png_header(1, 1))
        dims = get_png_dimensions(data)
        assert dims is not None
        assert dims.width_px == 1
        assert dims.height_px == 1

    def test_too_short_returns_none(self) -> None:
        short = _b64(b"\x89PNG\r\n\x1a\n\x00\x00")
        assert get_png_dimensions(short) is None

    def test_wrong_signature_returns_none(self) -> None:
        bad_sig = _b64(b"\x00\x00\x00\x00" + b"\x00" * 20)
        assert get_png_dimensions(bad_sig) is None

    def test_invalid_base64_returns_none(self) -> None:
        assert get_png_dimensions("!!!not-valid-b64!!!") is None

    def test_empty_string_returns_none(self) -> None:
        assert get_png_dimensions("") is None


# ---------------------------------------------------------------------------
# get_jpeg_dimensions
# ---------------------------------------------------------------------------


class TestGetJpegDimensions:
    def test_valid_jpeg(self) -> None:
        data = _b64(_make_jpeg_data(640, 480))
        dims = get_jpeg_dimensions(data)
        assert dims is not None
        assert dims.width_px == 640
        assert dims.height_px == 480

    def test_small_jpeg(self) -> None:
        data = _b64(_make_jpeg_data(1, 1))
        dims = get_jpeg_dimensions(data)
        assert dims is not None
        assert dims.width_px == 1
        assert dims.height_px == 1

    def test_too_short_returns_none(self) -> None:
        assert get_jpeg_dimensions(_b64(b"\xff")) is None

    def test_wrong_signature_returns_none(self) -> None:
        assert get_jpeg_dimensions(_b64(b"\x00\x00" + b"\x00" * 30)) is None

    def test_no_sof_marker_returns_none(self) -> None:
        # SOI followed by only non-SOF markers, then truncated
        data = b"\xff\xd8\xff\xe1\x00\x02"
        assert get_jpeg_dimensions(_b64(data)) is None

    def test_invalid_base64_returns_none(self) -> None:
        assert get_jpeg_dimensions("!!!bad!!!") is None

    def test_empty_string_returns_none(self) -> None:
        assert get_jpeg_dimensions("") is None


# ---------------------------------------------------------------------------
# get_gif_dimensions
# ---------------------------------------------------------------------------


class TestGetGifDimensions:
    def test_valid_gif89a(self) -> None:
        data = _b64(_make_gif_data(100, 50))
        dims = get_gif_dimensions(data)
        assert dims is not None
        assert dims.width_px == 100
        assert dims.height_px == 50

    def test_valid_gif87a(self) -> None:
        raw = b"GIF87a" + struct.pack("<H", 200) + struct.pack("<H", 150)
        dims = get_gif_dimensions(_b64(raw))
        assert dims is not None
        assert dims.width_px == 200
        assert dims.height_px == 150

    def test_too_short_returns_none(self) -> None:
        assert get_gif_dimensions(_b64(b"GIF89a\x00")) is None

    def test_wrong_signature_returns_none(self) -> None:
        assert get_gif_dimensions(_b64(b"NOTGIF" + b"\x00" * 10)) is None

    def test_empty_returns_none(self) -> None:
        assert get_gif_dimensions("") is None


# ---------------------------------------------------------------------------
# get_webp_dimensions
# ---------------------------------------------------------------------------


class TestGetWebpDimensions:
    def test_valid_vp8_lossy(self) -> None:
        data = _b64(_make_webp_vp8_data(800, 600))
        dims = get_webp_dimensions(data)
        assert dims is not None
        assert dims.width_px == 800
        assert dims.height_px == 600

    def test_too_short_returns_none(self) -> None:
        assert get_webp_dimensions(_b64(b"RIFF" + b"\x00" * 10)) is None

    def test_wrong_signature_returns_none(self) -> None:
        assert get_webp_dimensions(_b64(b"\x00" * 30)) is None

    def test_empty_returns_none(self) -> None:
        assert get_webp_dimensions("") is None


# ---------------------------------------------------------------------------
# get_image_dimensions (dispatch)
# ---------------------------------------------------------------------------


class TestGetImageDimensions:
    def test_dispatches_png(self) -> None:
        data = _b64(_make_png_header(10, 20))
        dims = get_image_dimensions(data, "image/png")
        assert dims is not None
        assert dims.width_px == 10

    def test_dispatches_jpeg(self) -> None:
        data = _b64(_make_jpeg_data(30, 40))
        dims = get_image_dimensions(data, "image/jpeg")
        assert dims is not None
        assert dims.width_px == 30

    def test_dispatches_gif(self) -> None:
        data = _b64(_make_gif_data(50, 60))
        dims = get_image_dimensions(data, "image/gif")
        assert dims is not None
        assert dims.width_px == 50

    def test_dispatches_webp(self) -> None:
        data = _b64(_make_webp_vp8_data(70, 80))
        dims = get_image_dimensions(data, "image/webp")
        assert dims is not None
        assert dims.width_px == 70

    def test_unknown_mime_returns_none(self) -> None:
        assert get_image_dimensions("AAAA", "image/bmp") is None


# ---------------------------------------------------------------------------
# image_fallback
# ---------------------------------------------------------------------------


class TestImageFallback:
    def test_mime_only(self) -> None:
        result = image_fallback("image/png")
        assert result == "[Image: [image/png]]"

    def test_with_dimensions(self) -> None:
        dims = ImageDimensions(width_px=800, height_px=600)
        result = image_fallback("image/png", dimensions=dims)
        assert result == "[Image: [image/png] 800x600]"

    def test_with_filename(self) -> None:
        result = image_fallback("image/jpeg", filename="photo.jpg")
        assert result == "[Image: photo.jpg [image/jpeg]]"

    def test_with_filename_and_dimensions(self) -> None:
        dims = ImageDimensions(width_px=1920, height_px=1080)
        result = image_fallback("image/png", dimensions=dims, filename="screen.png")
        assert result == "[Image: screen.png [image/png] 1920x1080]"

    def test_no_dimensions_no_filename(self) -> None:
        result = image_fallback("image/gif")
        assert "[image/gif]" in result


# ---------------------------------------------------------------------------
# allocate_image_id
# ---------------------------------------------------------------------------


class TestAllocateImageId:
    def test_returns_positive_int(self) -> None:
        img_id = allocate_image_id()
        assert isinstance(img_id, int)
        assert img_id >= 1

    def test_within_valid_range(self) -> None:
        for _ in range(100):
            img_id = allocate_image_id()
            assert 1 <= img_id <= 0xFFFFFFFE

    def test_returns_different_ids(self) -> None:
        ids = {allocate_image_id() for _ in range(50)}
        # With a 32-bit range, 50 calls should produce at least 2 unique values
        assert len(ids) > 1


# ---------------------------------------------------------------------------
# encode_kitty
# ---------------------------------------------------------------------------


class TestEncodeKitty:
    def test_small_payload_single_chunk(self) -> None:
        data = _b64(b"tiny")
        result = encode_kitty(data)
        assert result.startswith("\x1b_G")
        assert result.endswith("\x1b\\")
        assert "a=T" in result
        assert "f=100" in result
        assert "q=2" in result
        assert data in result

    def test_no_chunking_marker_for_small_data(self) -> None:
        data = _b64(b"small")
        result = encode_kitty(data)
        assert "m=" not in result

    def test_with_columns_and_rows(self) -> None:
        data = _b64(b"x")
        result = encode_kitty(data, columns=40, rows=10)
        assert "c=40" in result
        assert "r=10" in result

    def test_with_image_id(self) -> None:
        data = _b64(b"x")
        result = encode_kitty(data, image_id=42)
        assert "i=42" in result

    def test_large_payload_chunked(self) -> None:
        # Create data larger than 4096 base64 chars
        raw = b"A" * 4000  # base64 will be > 5000 chars
        data = _b64(raw)
        assert len(data) > 4096
        result = encode_kitty(data)
        # First chunk has m=1
        assert "m=1" in result
        # Last chunk has m=0
        assert "m=0" in result

    def test_large_payload_contains_all_data(self) -> None:
        raw = b"B" * 4000
        data = _b64(raw)
        result = encode_kitty(data)
        # All base64 chars should be present (split across chunks)
        # Strip control sequences and check
        payload = ""
        for part in result.split("\x1b\\"):
            if ";" in part:
                payload += part.split(";", 1)[-1]
        assert payload == data

    def test_none_params_omitted(self) -> None:
        data = _b64(b"x")
        result = encode_kitty(data, columns=None, rows=None, image_id=None)
        assert "c=" not in result
        assert "r=" not in result
        assert "i=" not in result


# ---------------------------------------------------------------------------
# encode_iterm2
# ---------------------------------------------------------------------------


class TestEncodeIterm2:
    def test_basic_inline(self) -> None:
        data = _b64(b"img")
        result = encode_iterm2(data)
        assert result.startswith("\x1b]1337;File=")
        assert result.endswith("\x07")
        assert "inline=1" in result
        assert f":{data}\x07" in result

    def test_with_width_and_height(self) -> None:
        result = encode_iterm2(_b64(b"x"), width=80, height="auto")
        assert "width=80" in result
        assert "height=auto" in result

    def test_with_name(self) -> None:
        result = encode_iterm2(_b64(b"x"), name="photo.png")
        name_b64 = base64.b64encode(b"photo.png").decode()
        assert f"name={name_b64}" in result

    def test_preserve_aspect_ratio_false(self) -> None:
        result = encode_iterm2(_b64(b"x"), preserve_aspect_ratio=False)
        assert "preserveAspectRatio=0" in result

    def test_preserve_aspect_ratio_none_omitted(self) -> None:
        result = encode_iterm2(_b64(b"x"))
        assert "preserveAspectRatio" not in result

    def test_not_inline(self) -> None:
        result = encode_iterm2(_b64(b"x"), inline=False)
        assert "inline=0" in result


# ---------------------------------------------------------------------------
# delete_kitty_image / delete_all_kitty_images
# ---------------------------------------------------------------------------


class TestDeleteKittyImage:
    def test_delete_specific_image(self) -> None:
        result = delete_kitty_image(99)
        assert result == "\x1b_Ga=d,d=I,i=99\x1b\\"

    def test_delete_all_images(self) -> None:
        result = delete_all_kitty_images()
        assert result == "\x1b_Ga=d,d=A\x1b\\"


# ---------------------------------------------------------------------------
# is_image_line
# ---------------------------------------------------------------------------


class TestIsImageLine:
    def test_kitty_prefix(self) -> None:
        assert is_image_line("\x1b_Ga=T;data\x1b\\") is True

    def test_iterm2_prefix(self) -> None:
        assert is_image_line("\x1b]1337;File=inline=1:data\x07") is True

    def test_kitty_embedded(self) -> None:
        assert is_image_line("some text \x1b_Gdata\x1b\\") is True

    def test_iterm2_embedded(self) -> None:
        assert is_image_line("prefix\x1b]1337;File=x") is True

    def test_plain_text(self) -> None:
        assert is_image_line("hello world") is False

    def test_empty_string(self) -> None:
        assert is_image_line("") is False


# ---------------------------------------------------------------------------
# reset_capabilities_cache
# ---------------------------------------------------------------------------


class TestResetCapabilitiesCache:
    def test_cache_cleared(self) -> None:
        # Populate cache first
        _ = get_capabilities()
        reset_capabilities_cache()
        # After reset, the internal cache should be None
        from pi.tui import terminal_image

        assert terminal_image._cached_capabilities is None

    def test_get_capabilities_repopulates_after_reset(self) -> None:
        reset_capabilities_cache()
        caps = get_capabilities()
        assert isinstance(caps, TerminalCapabilities)


# ---------------------------------------------------------------------------
# calculate_image_rows
# ---------------------------------------------------------------------------


class TestCalculateImageRows:
    def test_basic_calculation(self) -> None:
        dims = ImageDimensions(width_px=180, height_px=180)
        cell = CellDimensions(width_px=9, height_px=18)
        rows = calculate_image_rows(dims, target_width_cells=20, cell_dims=cell)
        # 20 cells * 9px = 180px target width
        # scale = 180/180 = 1.0
        # scaled_height = 180 * 1.0 = 180px
        # rows = ceil(180 / 18) = 10
        assert rows == 10

    def test_wide_image_scaled_down(self) -> None:
        dims = ImageDimensions(width_px=360, height_px=180)
        cell = CellDimensions(width_px=9, height_px=18)
        rows = calculate_image_rows(dims, target_width_cells=20, cell_dims=cell)
        # target_width_px = 20*9 = 180
        # scale = 180/360 = 0.5
        # scaled_height = 180*0.5 = 90
        # rows = ceil(90/18) = 5
        assert rows == 5

    def test_tall_image(self) -> None:
        dims = ImageDimensions(width_px=90, height_px=360)
        cell = CellDimensions(width_px=9, height_px=18)
        rows = calculate_image_rows(dims, target_width_cells=10, cell_dims=cell)
        # target_width_px = 10*9 = 90
        # scale = 90/90 = 1.0
        # scaled_height = 360
        # rows = ceil(360/18) = 20
        assert rows == 20

    def test_minimum_one_row(self) -> None:
        dims = ImageDimensions(width_px=1000, height_px=1)
        cell = CellDimensions(width_px=9, height_px=18)
        rows = calculate_image_rows(dims, target_width_cells=10, cell_dims=cell)
        assert rows >= 1

    def test_default_cell_dims_used_when_none(self) -> None:
        dims = ImageDimensions(width_px=180, height_px=180)
        rows = calculate_image_rows(dims, target_width_cells=20, cell_dims=None)
        # Default cell_dims: 9x18, same as explicit test above
        assert rows == 10

    def test_fractional_row_rounds_up(self) -> None:
        # 9*10 = 90 px target
        # scale = 90/100 = 0.9
        # scaled_height = 100*0.9 = 90
        # rows = ceil(90/18) = 5.0 = 5
        dims = ImageDimensions(width_px=100, height_px=100)
        cell = CellDimensions(width_px=9, height_px=18)
        rows = calculate_image_rows(dims, target_width_cells=10, cell_dims=cell)
        assert rows == 5

    def test_non_even_division_rounds_up(self) -> None:
        dims = ImageDimensions(width_px=100, height_px=100)
        cell = CellDimensions(width_px=10, height_px=10)
        # target = 8*10 = 80
        # scale = 80/100 = 0.8
        # scaled_height = 100*0.8 = 80
        # rows = ceil(80/10) = 8
        rows = calculate_image_rows(dims, target_width_cells=8, cell_dims=cell)
        assert rows == 8

    def test_rounding_up_with_remainder(self) -> None:
        dims = ImageDimensions(width_px=100, height_px=55)
        cell = CellDimensions(width_px=10, height_px=10)
        # target = 10*10 = 100
        # scale = 100/100 = 1.0
        # scaled_height = 55
        # rows = ceil(55/10) = 6
        rows = calculate_image_rows(dims, target_width_cells=10, cell_dims=cell)
        assert rows == 6


# ---------------------------------------------------------------------------
# Cell dimensions get/set
# ---------------------------------------------------------------------------


class TestCellDimensions:
    def test_get_default_cell_dimensions(self) -> None:
        dims = get_cell_dimensions()
        assert dims.width_px == 9
        assert dims.height_px == 18

    def test_set_and_get_cell_dimensions(self) -> None:
        original = get_cell_dimensions()
        try:
            set_cell_dimensions(CellDimensions(width_px=12, height_px=24))
            dims = get_cell_dimensions()
            assert dims.width_px == 12
            assert dims.height_px == 24
        finally:
            # Restore original to avoid leaking state
            set_cell_dimensions(original)


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_image_dimensions(self) -> None:
        d = ImageDimensions(width_px=100, height_px=200)
        assert d.width_px == 100
        assert d.height_px == 200

    def test_cell_dimensions(self) -> None:
        d = CellDimensions(width_px=9, height_px=18)
        assert d.width_px == 9
        assert d.height_px == 18

    def test_terminal_capabilities(self) -> None:
        c = TerminalCapabilities(images="kitty", true_color=True, hyperlinks=True)
        assert c.images == "kitty"
        assert c.true_color is True
        assert c.hyperlinks is True

    def test_image_render_options_defaults(self) -> None:
        o = ImageRenderOptions()
        assert o.max_width_cells is None
        assert o.max_height_cells is None
        assert o.preserve_aspect_ratio is None
        assert o.image_id is None

    def test_image_render_options_custom(self) -> None:
        o = ImageRenderOptions(
            max_width_cells=80,
            max_height_cells=40,
            preserve_aspect_ratio=True,
            image_id=7,
        )
        assert o.max_width_cells == 80
        assert o.max_height_cells == 40
        assert o.preserve_aspect_ratio is True
        assert o.image_id == 7
