
"""
DriveGuard AI
PDF report + GPS + Evidence Photo + Static Map
"""

import io
import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

from flask import Blueprint, abort, jsonify, request, send_file

BASE = Path(__file__).resolve().parent
DB = BASE / "drowsiness.db"
EVIDENCE_DIR = BASE / "private_evidence"

SAVE_EVIDENCE = os.getenv(
    "DRIVEGUARD_SAVE_EVIDENCE", "0"
) == "1"

RETENTION_DAYS = max(
    1,
    int(os.getenv("DRIVEGUARD_RETENTION_DAYS", "7"))
)

MAX_IMAGES_PDF = 12
MAX_ROWS_PDF = 2000

bp = Blueprint("evidence_reports", __name__)


# ==========================================
# DATABASE
# ==========================================

def _connect():
    conn = sqlite3.connect(DB, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def upgrade_schema():
    EVIDENCE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                ear REAL,
                mar REAL,
                created_at TEXT NOT NULL
            )
        """)

        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(alerts)"
            )
        }

        additions = {
            "evidence_path": "TEXT",
            "latitude": "REAL",
            "longitude": "REAL",
            "location_accuracy": "REAL",
            "location_recorded_at": "TEXT"
        }

        for name, data_type in additions.items():
            if name not in columns:
                conn.execute(
                    f"ALTER TABLE alerts "
                    f"ADD COLUMN {name} {data_type}"
                )

    cleanup_expired()


def _image_path(alert_id, stored_name):
    expected = f"alert_{int(alert_id)}.jpg"

    if stored_name != expected:
        return None

    return EVIDENCE_DIR / expected


def _local_only():
    if request.remote_addr not in (
        "127.0.0.1", "::1"
    ):
        abort(403)


# ==========================================
# XOA ANH HET HAN
# ==========================================

def cleanup_expired():
    cutoff = datetime.now() - timedelta(
        days=RETENTION_DAYS
    )

    with _connect() as conn:
        rows = conn.execute("""
            SELECT id, evidence_path, created_at
            FROM alerts
            WHERE evidence_path IS NOT NULL
        """).fetchall()

        for row in rows:
            try:
                when = datetime.fromisoformat(
                    str(row["created_at"])
                )

                if when.tzinfo is not None:
                    when = (
                        when.astimezone()
                        .replace(tzinfo=None)
                    )
            except (ValueError, TypeError):
                continue

            if when < cutoff:
                path = _image_path(
                    row["id"],
                    row["evidence_path"]
                )

                if path:
                    path.unlink(missing_ok=True)

                conn.execute("""
                    UPDATE alerts
                    SET evidence_path = NULL
                    WHERE id = ?
                """, (row["id"],))


# ==========================================
# LUU CANH BAO + GPS + ANH
# ==========================================

def save_alert_with_evidence(
    event_type,
    ear=None,
    mar=None,
    frame=None
):
    if event_type not in (
        "DROWSY", "YAWNING"
    ):
        raise ValueError(
            "Loai canh bao khong hop le"
        )

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    # Lay GPS moi nhat tai thoi diem canh bao
    from location_bridge import get_recent_location

    location = get_recent_location()

    latitude = (
        location["latitude"] if location else None
    )

    longitude = (
        location["longitude"] if location else None
    )

    accuracy = (
        location["accuracy"] if location else None
    )

    gps_time = (
        location["recorded_at"] if location else None
    )

    with _connect() as conn:
        cursor = conn.execute("""
            INSERT INTO alerts (
                event_type,
                ear,
                mar,
                created_at,
                latitude,
                longitude,
                location_accuracy,
                location_recorded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_type,
            ear,
            mar,
            timestamp,
            latitude,
            longitude,
            accuracy,
            gps_time
        ))

        alert_id = cursor.lastrowid

        if SAVE_EVIDENCE and frame is not None:
            try:
                import cv2
                import numpy as np

                image = frame.copy()
                height, width = image.shape[:2]

                # Bang thong tin nam duoi anh
                info_band = np.zeros(
                    (90, width, 3),
                    dtype=np.uint8
                )

                image = np.vstack(
                    (image, info_band)
                )

                cv2.putText(
                    image,
                    f"ALERT #{alert_id} | "
                    f"{event_type} | {timestamp}",
                    (10, height + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                if location:
                    gps_text = (
                        f"LOCATION: {latitude:.6f}, "
                        f"{longitude:.6f}"
                    )

                    accuracy_text = (
                        f"Estimated accuracy: "
                        f"+/- {accuracy:.1f} m"
                    )
                else:
                    gps_text = (
                        "LOCATION: NOT AVAILABLE"
                    )

                    accuracy_text = (
                        "No recent location fix"
                    )

                cv2.putText(
                    image,
                    gps_text,
                    (10, height + 46),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                cv2.putText(
                    image,
                    accuracy_text,
                    (10, height + 69),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA
                )

                h, w = image.shape[:2]

                scale = min(
                    1.0,
                    640.0 / max(w, 1)
                )

                if scale < 1:
                    image = cv2.resize(
                        image,
                        (
                            max(1, round(w * scale)),
                            max(1, round(h * scale))
                        )
                    )

                ok, encoded = cv2.imencode(
                    ".jpg",
                    image,
                    [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        80
                    ]
                )

                if ok:
                    EVIDENCE_DIR.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    filename = (
                        f"alert_{alert_id}.jpg"
                    )

                    path = EVIDENCE_DIR / filename

                    path.write_bytes(
                        encoded.tobytes()
                    )

                    conn.execute("""
                        UPDATE alerts
                        SET evidence_path = ?
                        WHERE id = ?
                    """, (
                        filename,
                        alert_id
                    ))

            except Exception as exc:
                print(
                    "[EVIDENCE] Loi luu anh:",
                    exc
                )

    return alert_id


# ==========================================
# GOOGLE MAPS
# ==========================================

def _google_maps_url(lat, lng):
    return (
        "https://www.google.com/maps/search/"
        f"?api=1&query={float(lat):.6f},"
        f"{float(lng):.6f}"
    )


# ==========================================
# TAI ANH BAN DO TINH
# ==========================================


def _download_static_map(lat, lng):
    """
    Tao ban do thu nho tu OpenStreetMap.
    Danh dau GPS va vong tron sai so.
    Chi tai cac o can cho anh hien tai.
    Cache cac o ban do trong 7 ngay.
    """

    import math
    import time
    import requests

    from PIL import (
        Image as PILImage,
        ImageDraw,
        ImageFont
    )

    # Cau hinh
    zoom = 15
    width = 520
    height = 300
    tile_size = 256

    lat = float(lat)
    lng = float(lng)

    # Gioi han Mercator
    lat = max(-85.0511, min(85.0511, lat))
    lng = max(-180.0, min(180.0, lng))

    # Thu muc cache
    cache_dir = BASE / "map_tile_cache"
    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cache_seconds = 7 * 24 * 60 * 60

    # Chuyen toa do sang pixel Web Mercator
    scale = 2 ** zoom

    lat_rad = math.radians(lat)

    center_x = (
        (lng + 180.0) / 360.0
        * scale * tile_size
    )

    center_y = (
        (
            1.0 -
            math.asinh(math.tan(lat_rad)) / math.pi
        ) / 2.0
        * scale * tile_size
    )

    left = center_x - width / 2
    top = center_y - height / 2

    first_x = math.floor(left / tile_size)
    first_y = math.floor(top / tile_size)

    last_x = math.floor(
        (left + width - 1) / tile_size
    )

    last_y = math.floor(
        (top + height - 1) / tile_size
    )

    # Anh ban do
    result = PILImage.new(
        "RGB",
        (width, height),
        (235, 240, 245)
    )

    session = requests.Session()

    session.headers.update({
        "User-Agent": (
            "DriveGuardAI-LocalResearch/1.0 "
            "(local PDF map preview)"
        ),
        "Accept": "image/png"
    })

    tiles_loaded = 0

    for tile_x in range(first_x, last_x + 1):
        for tile_y in range(first_y, last_y + 1):

            if not (0 <= tile_y < scale):
                continue

            wrapped_x = tile_x % scale

            cache_file = cache_dir / (
                f"{zoom}_{wrapped_x}_{tile_y}.png"
            )

            tile = None

            # Doc cache neu con han
            if cache_file.exists():
                age = (
                    time.time()
                    - cache_file.stat().st_mtime
                )

                if age < cache_seconds:
                    try:
                        with PILImage.open(
                            cache_file
                        ) as cached:
                            tile = cached.convert("RGB")
                    except Exception:
                        tile = None

            # Neu chua co cache thi tai tile
            if tile is None:
                url = (
                    "https://tile.openstreetmap.org/"
                    f"{zoom}/{wrapped_x}/{tile_y}.png"
                )

                try:
                    response = session.get(
                        url,
                        timeout=8
                    )

                    response.raise_for_status()

                    with PILImage.open(
                        io.BytesIO(response.content)
                    ) as downloaded:
                        tile = downloaded.convert("RGB")

                    tile.save(
                        cache_file,
                        format="PNG"
                    )

                except Exception as exc:
                    print(
                        "[GPS MAP] Loi tai tile:",
                        url,
                        exc
                    )
                    continue

            paste_x = round(
                tile_x * tile_size - left
            )

            paste_y = round(
                tile_y * tile_size - top
            )

            result.paste(
                tile,
                (paste_x, paste_y)
            )

            tiles_loaded += 1

    session.close()

    if tiles_loaded == 0:
        return None

    draw = ImageDraw.Draw(result)

    center = (width // 2, height // 2)

    # Tinh so met tren moi pixel tai vi do
    meters_per_pixel = (
        156543.03392
        * math.cos(lat_rad)
        / scale
    )

    # Ban kinh sai so lay tu ban ghi GPS
    # se duoc truyen qua tham so neu co.
    # Mac dinh chi ve marker khi chua co sai so.

    # Vong trang bao quanh marker
    draw.ellipse(
        (
            center[0] - 15,
            center[1] - 15,
            center[0] + 15,
            center[1] + 15
        ),
        fill=(255, 255, 255),
        outline=(220, 35, 45),
        width=3
    )

    # Marker do
    draw.ellipse(
        (
            center[0] - 9,
            center[1] - 9,
            center[0] + 9,
            center[1] + 9
        ),
        fill=(220, 35, 45),
        outline=(255, 255, 255),
        width=2
    )

    # Thong tin toa do
    coordinate_text = (
        f"GPS: {lat:.6f}, {lng:.6f}"
    )

    draw.rectangle(
        (0, 0, width, 30),
        fill=(11, 30, 50)
    )

    draw.text(
        (12, 9),
        coordinate_text,
        fill=(255, 255, 255)
    )

    # Ghi cong nguon du lieu ban do
    draw.rectangle(
        (0, height - 23, width, height),
        fill=(255, 255, 255)
    )

    draw.text(
        (8, height - 17),
        "Map data: (c) OpenStreetMap contributors",
        fill=(30, 45, 60)
    )

    output = io.BytesIO()

    result.save(
        output,
        format="PNG"
    )

    output.seek(0)

    print(
        "[GPS MAP] Da tao ban do:",
        lat,
        lng,
        "| Tiles:",
        tiles_loaded
    )

    return output

# ==========================================
# BO LOC BAO CAO
# ==========================================

def _parse_filters():
    start = request.args.get(
        "from", ""
    ).strip()

    end = request.args.get(
        "to", ""
    ).strip()

    single = request.args.get(
        "date", ""
    ).strip()

    kind = request.args.get(
        "type", ""
    ).strip()

    if single and not start and not end:
        start = end = single

    if kind not in (
        "", "DROWSY", "YAWNING"
    ):
        abort(
            400,
            description="Loai canh bao khong hop le"
        )

    try:
        d1 = (
            date.fromisoformat(start)
            if start else None
        )

        d2 = (
            date.fromisoformat(end)
            if end else None
        )

    except ValueError:
        abort(
            400,
            description="Ngay khong hop le"
        )

    if d1 and d2 and d1 > d2:
        abort(
            400,
            description="Khoang ngay khong hop le"
        )

    conditions = []
    params = []

    if d1:
        conditions.append(
            "date(created_at) >= ?"
        )
        params.append(d1.isoformat())

    if d2:
        conditions.append(
            "date(created_at) <= ?"
        )
        params.append(d2.isoformat())

    if kind:
        conditions.append(
            "event_type = ?"
        )
        params.append(kind)

    where_sql = (
        " WHERE " + " AND ".join(conditions)
        if conditions else ""
    )

    return where_sql, params, d1, d2, kind


# ==========================================
# FONT PDF
# ==========================================

def _pdf_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    fonts = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path(
            "/usr/share/fonts/truetype/"
            "dejavu/DejaVuSans.ttf"
        )
    ]

    for path in fonts:
        if path.is_file():
            name = "DriveGuardUnicode"

            if name not in (
                pdfmetrics.getRegisteredFontNames()
            ):
                pdfmetrics.registerFont(
                    TTFont(
                        name,
                        str(path)
                    )
                )

            return name

    raise RuntimeError(
        "Khong tim thay font Unicode"
    )


def _fmt_number(value):
    return (
        "-"
        if value is None
        else f"{float(value):.3f}"
    )


def _footer(canvas, doc):
    from reportlab.lib.units import mm

    canvas.saveState()

    canvas.setFont(
        "Helvetica",
        8
    )

    canvas.setFillColorRGB(
        0.42, 0.48, 0.54
    )

    canvas.drawString(
        18 * mm,
        12 * mm,
        "DriveGuard AI | Bao cao nghien cuu"
    )

    canvas.drawRightString(
        192 * mm,
        12 * mm,
        f"Trang {doc.page}"
    )

    canvas.restoreState()


# ==========================================
# XUAT PDF
# ==========================================

@bp.get("/api/export/pdf")
def export_pdf():
    _local_only()

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader

    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image,
        PageBreak,
        KeepTogether
    )

    from reportlab.graphics.shapes import (
        Drawing,
        Rect,
        String
    )

    where, params, d1, d2, kind = (
        _parse_filters()
    )

    with _connect() as conn:
        rows = conn.execute("""
            SELECT
                id,
                event_type,
                ear,
                mar,
                created_at,
                evidence_path,
                latitude,
                longitude,
                location_accuracy,
                location_recorded_at
            FROM alerts
        """ + where + """
            ORDER BY id DESC
        """, params).fetchall()

    if len(rows) > MAX_ROWS_PDF:
        abort(
            413,
            description=(
                "Qua nhieu ban ghi, "
                "hay loc theo ngay"
            )
        )

    font = _pdf_font()

    navy = colors.HexColor("#0B1E32")
    teal = colors.HexColor("#12BBD0")
    light = colors.HexColor("#E5F5F8")
    border = colors.HexColor("#D9E1E7")

    normal = ParagraphStyle(
        "DG_Normal",
        fontName=font,
        fontSize=9,
        leading=14,
        textColor=navy
    )

    title = ParagraphStyle(
        "DG_Title",
        parent=normal,
        fontSize=17,
        leading=23,
        spaceAfter=8
    )

    heading = ParagraphStyle(
        "DG_Heading",
        parent=normal,
        fontSize=12,
        leading=17,
        spaceBefore=10,
        spaceAfter=7
    )

    small = ParagraphStyle(
        "DG_Small",
        parent=normal,
        fontSize=7.5,
        leading=11
    )

    center = ParagraphStyle(
        "DG_Center",
        parent=small,
        alignment=1
    )

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=19 * mm
    )

    story = []

    # =====================================
    # TONG QUAN
    # =====================================

    total = len(rows)

    drowsy = sum(
        r["event_type"] == "DROWSY"
        for r in rows
    )

    yawning = sum(
        r["event_type"] == "YAWNING"
        for r in rows
    )

    rate = (
        drowsy / total * 100
        if total else 0
    )

    story.append(
        Paragraph(
            "DRIVEGUARD AI - BÁO CÁO GIÁM SÁT",
            title
        )
    )

    story.append(
        Paragraph(
            "Ngày xuất: " +
            datetime.now().strftime(
                "%d/%m/%Y %H:%M:%S"
            ),
            normal
        )
    )

    period = (
        f"{d1.isoformat() if d1 else 'Tất cả'} "
        f"đến "
        f"{d2.isoformat() if d2 else 'Tất cả'}"
    )

    kind_name = {
        "": "Tất cả",
        "DROWSY": "Buồn ngủ",
        "YAWNING": "Ngáp"
    }[kind]

    story.append(
        Paragraph(
            f"Phạm vi: {period} | Loại: {kind_name}",
            normal
        )
    )

    story.append(
        Spacer(1, 6 * mm)
    )

    summary = Table(
        [
            [
                "TỔNG CẢNH BÁO",
                "BUỒN NGỦ",
                "NGÁP",
                "TỶ LỆ BUỒN NGỦ"
            ],
            [
                str(total),
                str(drowsy),
                str(yawning),
                f"{rate:.1f}%"
            ]
        ],
        colWidths=[43.5 * mm] * 4,
        rowHeights=[
            11 * mm,
            15 * mm
        ]
    )

    summary.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                light
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                font
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, 0),
                8
            ),
            (
                "FONTSIZE",
                (0, 1),
                (-1, 1),
                16
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LINEBELOW",
                (0, 1),
                (-1, 1),
                1,
                teal
            )
        ])
    )

    story.append(summary)
    story.append(
        Spacer(1, 5 * mm)
    )

    # =====================================
    # BIEU DO
    # =====================================

    story.append(
        Paragraph(
            "PHÂN BỐ CẢNH BÁO",
            heading
        )
    )

    chart = Drawing(
        470,
        72
    )

    maximum = max(
        drowsy,
        yawning,
        1
    )

    chart_data = [
        (
            "Buồn ngủ",
            drowsy,
            colors.HexColor("#E96780")
        ),
        (
            "Ngáp",
            yawning,
            colors.HexColor("#F5B72E")
        )
    ]

    for index, (
        name,
        value,
        bar_color
    ) in enumerate(chart_data):

        y = 42 - index * 34

        chart.add(
            String(
                0,
                y + 4,
                name,
                fontName=font,
                fontSize=9
            )
        )

        chart.add(
            Rect(
                100,
                y,
                290,
                14,
                fillColor=light,
                strokeColor=None
            )
        )

        chart.add(
            Rect(
                100,
                y,
                290 * value / maximum,
                14,
                fillColor=bar_color,
                strokeColor=None
            )
        )

        chart.add(
            String(
                405,
                y + 4,
                str(value),
                fontName=font,
                fontSize=10
            )
        )

    story.append(chart)

    story.append(
        Paragraph(
            "Lưu ý: Các số liệu là sự kiện "
            "phần mềm ghi nhận, không phải "
            "độ chính xác nhận diện đã xác minh.",
            small
        )
    )

    # =====================================
    # BANG LICH SU CANH BAO
    # =====================================

    story.append(
        Paragraph(
            "CHI TIẾT LỊCH SỬ CẢNH BÁO",
            heading
        )
    )

    history_data = [[
        "ID",
        "Loại",
        "EAR",
        "MAR",
        "Thời điểm",
        "Ảnh"
    ]]

    for r in rows:
        path = _image_path(
            r["id"],
            r["evidence_path"]
        )

        has_image = bool(
            path and path.is_file()
        )

        history_data.append([
            str(r["id"]),
            (
                "Buồn ngủ"
                if r["event_type"] == "DROWSY"
                else "Ngáp"
            ),
            _fmt_number(r["ear"]),
            _fmt_number(r["mar"]),
            str(
                r["created_at"]
            ).replace("T", " ")[:19],
            "Có" if has_image else "-"
        ])

    history_table = Table(
        history_data,
        colWidths=[
            14 * mm,
            31 * mm,
            19 * mm,
            19 * mm,
            70 * mm,
            21 * mm
        ],
        repeatRows=1
    )

    history_table.setStyle(
        TableStyle([
            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                font
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                light
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#F6F9FC")
                ]
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.3,
                border
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                4
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5
            )
        ])
    )

    story.append(history_table)

    # =====================================
    # BANG GPS
    # =====================================

    story.append(
        Paragraph(
            "VỊ TRÍ THIẾT BỊ KHI CẢNH BÁO",
            heading
        )
    )

    gps_data = [[
        "ID",
        "Thời điểm GPS",
        "Vĩ độ",
        "Kinh độ",
        "Sai số",
        "Bản đồ"
    ]]

    valid_gps_count = 0

    for r in rows:
        lat = r["latitude"]
        lng = r["longitude"]

        if lat is None or lng is None:
            gps_data.append([
                str(r["id"]),
                "-",
                "Chưa có",
                "-",
                "-",
                "-"
            ])
            continue

        valid_gps_count += 1

        accuracy = r["location_accuracy"]

        accuracy_text = (
            f"±{float(accuracy):.0f} m"
            if accuracy is not None
            else "-"
        )

        map_url = _google_maps_url(
            lat,
            lng
        ).replace("&", "&amp;")

        gps_data.append([
            str(r["id"]),
            str(
                r["location_recorded_at"] or "-"
            ).replace("T", " ")[:19],
            f"{float(lat):.6f}",
            f"{float(lng):.6f}",
            accuracy_text,
            Paragraph(
                f'<link href="{map_url}" '
                f'color="blue">Google Maps</link>',
                small
            )
        ])

    gps_table = Table(
        gps_data,
        colWidths=[
            12 * mm,
            43 * mm,
            28 * mm,
            28 * mm,
            20 * mm,
            43 * mm
        ],
        repeatRows=1
    )

    gps_table.setStyle(
        TableStyle([
            (
                "FONTNAME",
                (0, 0),
                (-1, -1),
                font
            ),
            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                light
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.3,
                border
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                3
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                3
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5
            )
        ])
    )

    story.append(gps_table)

    story.append(
        Spacer(1, 3 * mm)
    )

    story.append(
        Paragraph(
            f"Có {valid_gps_count}/{total} "
            "cảnh báo được gắn vị trí. "
            "Tọa độ là vị trí ước tính "
            "của thiết bị mở Dashboard.",
            small
        )
    )

    # =====================================
    # ANH MINH CHUNG + BAN DO
    # =====================================

    if request.args.get("images") == "1":

        selected = []

        for r in rows:
            path = _image_path(
                r["id"],
                r["evidence_path"]
            )

            if path and path.is_file():
                selected.append(
                    (r, path)
                )

            if len(selected) >= MAX_IMAGES_PDF:
                break

        if selected:
            story.append(
                PageBreak()
            )

            story.append(
                Paragraph(
                    "ẢNH MINH CHỨNG CẢNH BÁO",
                    title
                )
            )

            story.append(
                Paragraph(
                    "Ảnh webcam và bản đồ vị trí "
                    "ước tính tại thời điểm cảnh báo. "
                    "Tối đa 12 ảnh minh chứng.",
                    small
                )
            )

            story.append(
                Spacer(1, 5 * mm)
            )

            # Cache ban do cho tung toa do
            map_cache = {}

            for index, (r, path) in enumerate(
                selected
            ):
                if index > 0 and index % 2 == 0:
                    story.append(
                        PageBreak()
                    )

                label = (
                    "Buồn ngủ"
                    if r["event_type"] == "DROWSY"
                    else "Ngáp"
                )

                # -------------------------
                # ANH TAI XE
                # -------------------------

                img_w, img_h = ImageReader(
                    str(path)
                ).getSize()

                max_w = 78 * mm
                max_h = 55 * mm

                factor = min(
                    max_w / img_w,
                    max_h / img_h
                )

                photo = Image(
                    str(path),
                    width=img_w * factor,
                    height=img_h * factor
                )

                photo_caption = Paragraph(
                    f"<b>#{r['id']} - {label}</b>"
                    f"<br/>"
                    f"{escape(str(r['created_at']).replace('T', ' '))}"
                    f"<br/>EAR: {_fmt_number(r['ear'])}"
                    f" | MAR: {_fmt_number(r['mar'])}",
                    center
                )

                left_content = [
                    Paragraph(
                        "<b>Ảnh minh chứng tài xế</b>",
                        center
                    ),
                    Spacer(1, 3 * mm),
                    photo,
                    Spacer(1, 3 * mm),
                    photo_caption
                ]

                # -------------------------
                # BAN DO GPS
                # -------------------------

                lat = r["latitude"]
                lng = r["longitude"]
                accuracy = r["location_accuracy"]

                right_content = [
                    Paragraph(
                        "<b>Bản đồ vị trí cảnh báo</b>",
                        center
                    ),
                    Spacer(1, 3 * mm)
                ]

                if lat is not None and lng is not None:

                    lat = float(lat)
                    lng = float(lng)

                    cache_key = (
                        round(lat, 5),
                        round(lng, 5)
                    )

                    if cache_key not in map_cache:
                        map_buffer = (
                            _download_static_map(
                                lat,
                                lng
                            )
                        )

                        map_cache[cache_key] = (
                            map_buffer.getvalue()
                            if map_buffer else None
                        )

                    map_bytes = map_cache[cache_key]

                    if map_bytes:
                        map_image = Image(
                            io.BytesIO(map_bytes),
                            width=78 * mm,
                            height=45 * mm
                        )

                        right_content.append(
                            map_image
                        )

                        right_content.append(
                            Paragraph(
                                "Bản đồ: © OpenStreetMap "
                                "contributors",
                                center
                            )
                        )
                    else:
                        right_content.append(
                            Paragraph(
                                "Không tải được ảnh bản đồ. "
                                "Có thể xem vị trí bằng "
                                "liên kết bên dưới.",
                                center
                            )
                        )

                    right_content.append(
                        Spacer(1, 3 * mm)
                    )

                    gps_text = (
                        f"Vĩ độ: {lat:.6f}"
                        f"<br/>Kinh độ: {lng:.6f}"
                    )

                    if accuracy is not None:
                        gps_text += (
                            f"<br/>Sai số ước tính: "
                            f"±{float(accuracy):.1f} m"
                        )

                    gps_time = (
                        r["location_recorded_at"]
                    )

                    if gps_time:
                        gps_text += (
                            "<br/>Thời gian GPS: "
                            + escape(
                                str(gps_time)
                                .replace("T", " ")[:19]
                            )
                        )

                    map_url = _google_maps_url(
                        lat,
                        lng
                    ).replace("&", "&amp;")

                    gps_text += (
                        f'<br/><link href="{map_url}" '
                        'color="blue">'
                        'Xem trên Google Maps'
                        '</link>'
                    )

                    right_content.append(
                        Paragraph(
                            gps_text,
                            center
                        )
                    )

                else:
                    right_content.append(
                        Paragraph(
                            "Chưa có tọa độ GPS "
                            "tại thời điểm cảnh báo.",
                            center
                        )
                    )

                # -------------------------
                # GHEP ANH + BAN DO
                # -------------------------

                evidence_card = Table(
                    [[
                        left_content,
                        right_content
                    ]],
                    colWidths=[
                        87 * mm,
                        87 * mm
                    ]
                )

                evidence_card.setStyle(
                    TableStyle([
                        (
                            "BOX",
                            (0, 0),
                            (-1, -1),
                            0.5,
                            border
                        ),
                        (
                            "INNERGRID",
                            (0, 0),
                            (-1, -1),
                            0.4,
                            border
                        ),
                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "TOP"
                        ),
                        (
                            "ALIGN",
                            (0, 0),
                            (-1, -1),
                            "CENTER"
                        ),
                        (
                            "LEFTPADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        ),
                        (
                            "RIGHTPADDING",
                            (0, 0),
                            (-1, -1),
                            6
                        ),
                        (
                            "TOPPADDING",
                            (0, 0),
                            (-1, -1),
                            8
                        ),
                        (
                            "BOTTOMPADDING",
                            (0, 0),
                            (-1, -1),
                            8
                        )
                    ])
                )

                story.append(
                    KeepTogether([
                        evidence_card,
                        Spacer(1, 5 * mm)
                    ])
                )

    # =====================================
    # BUILD PDF
    # =====================================

    doc.build(
        story,
        onFirstPage=_footer,
        onLaterPages=_footer
    )

    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="driveguard_report.pdf"
    )


# =====================================
# XEM ANH MINH CHUNG
# ==========================================

@bp.get("/api/evidence/<int:alert_id>")
def get_evidence(alert_id):
    _local_only()

    with _connect() as conn:
        row = conn.execute("""
            SELECT evidence_path
            FROM alerts
            WHERE id = ?
        """, (alert_id,)).fetchone()

    if row is None:
        abort(404)

    path = _image_path(
        alert_id,
        row["evidence_path"]
    )

    if not path or not path.is_file():
        abort(404)

    response = send_file(
        path,
        mimetype="image/jpeg",
        as_attachment=True,
        download_name=path.name
    )

    response.headers[
        "Cache-Control"
    ] = "no-store"

    return response


# =====================================
# XOA ANH MINH CHUNG
# ==========================================

@bp.delete("/api/evidence/<int:alert_id>")
def delete_evidence(alert_id):
    _local_only()

    with _connect() as conn:
        row = conn.execute("""
            SELECT evidence_path
            FROM alerts
            WHERE id = ?
        """, (alert_id,)).fetchone()

        if row is None:
            abort(404)

        path = _image_path(
            alert_id,
            row["evidence_path"]
        )

        conn.execute("""
            UPDATE alerts
            SET evidence_path = NULL
            WHERE id = ?
        """, (alert_id,))

    if path:
        path.unlink(missing_ok=True)

    return jsonify({
        "success": True,
        "alert_id": alert_id
    })


# =====================================
# DANG KY FLASK
# =====================================

def install(app):
    upgrade_schema()

    if "evidence_reports" not in app.blueprints:
        app.register_blueprint(bp)
