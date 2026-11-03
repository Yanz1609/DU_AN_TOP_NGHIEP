
"""DriveGuard AI: PDF bao cao + anh minh chung (demo chi tren localhost)."""

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
SAVE_EVIDENCE = os.getenv("DRIVEGUARD_SAVE_EVIDENCE", "0") == "1"
RETENTION_DAYS = max(1, int(os.getenv("DRIVEGUARD_RETENTION_DAYS", "7")))
MAX_IMAGES_PDF = 12
MAX_ROWS_PDF = 2000

bp = Blueprint("evidence_reports", __name__)


# =====================================
# BAO VE API LOCALHOST
# =====================================

def _local_only():
    if request.remote_addr not in ("127.0.0.1", "::1"):
        abort(403)


# =====================================
# DATABASE
# =====================================

def _connect():
    db = sqlite3.connect(DB, timeout=15)
    db.row_factory = sqlite3.Row
    return db


def upgrade_schema():
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    with _connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                ear REAL,
                mar REAL,
                created_at TEXT NOT NULL
            )
        """)

        cols = {
            r["name"]
            for r in db.execute("PRAGMA table_info(alerts)")
        }

        if "evidence_path" not in cols:
            db.execute(
                "ALTER TABLE alerts ADD COLUMN evidence_path TEXT"
            )

    cleanup_expired()


def _image_path(alert_id, stored_name):
    """
    Chi doc anh do chinh module tao.
    Khong cho truy cap file ben ngoai.
    """
    expected = f"alert_{int(alert_id)}.jpg"

    if stored_name != expected:
        return None

    return EVIDENCE_DIR / expected


# =====================================
# XOA ANH HET HAN
# =====================================

def cleanup_expired():
    cutoff = datetime.now() - timedelta(days=RETENTION_DAYS)

    with _connect() as db:
        rows = db.execute("""
            SELECT id, evidence_path, created_at
            FROM alerts
            WHERE evidence_path IS NOT NULL
        """).fetchall()

        for row in rows:
            try:
                when = datetime.fromisoformat(
                    str(row["created_at"])
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

                db.execute(
                    """
                    UPDATE alerts
                    SET evidence_path = NULL
                    WHERE id = ?
                    """,
                    (row["id"],)
                )


# =====================================
# LUU CANH BAO VA ANH MINH CHUNG
# =====================================

def save_alert_with_evidence(
    event_type,
    ear=None,
    mar=None,
    frame=None
):
    if event_type not in ("DROWSY", "YAWNING"):
        raise ValueError("Loai canh bao khong hop le")

    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    with _connect() as db:
        cursor = db.execute("""
            INSERT INTO alerts(
                event_type,
                ear,
                mar,
                created_at
            )
            VALUES (?, ?, ?, ?)
        """, (
            event_type,
            ear,
            mar,
            timestamp
        ))

        alert_id = cursor.lastrowid

        if SAVE_EVIDENCE and frame is not None:
            try:
                import cv2

                height, width = frame.shape[:2]

                factor = min(
                    1.0,
                    640.0 / max(width, 1)
                )

                if factor < 1:
                    resized = cv2.resize(
                        frame,
                        (
                            max(1, round(width * factor)),
                            max(1, round(height * factor))
                        )
                    )
                else:
                    resized = frame

                ok, buf = cv2.imencode(
                    ".jpg",
                    resized,
                    [
                        int(cv2.IMWRITE_JPEG_QUALITY),
                        72
                    ]
                )

                if ok:
                    EVIDENCE_DIR.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    filename = f"alert_{alert_id}.jpg"
                    path = EVIDENCE_DIR / filename

                    path.write_bytes(buf.tobytes())

                    db.execute(
                        """
                        UPDATE alerts
                        SET evidence_path = ?
                        WHERE id = ?
                        """,
                        (filename, alert_id)
                    )

            except Exception as exc:
                print(
                    "CANH BAO: Khong luu duoc anh minh chung:",
                    exc
                )

    return alert_id


# =====================================
# BO LOC BAO CAO
# =====================================

def _parse_filters():
    start = request.args.get("from", "").strip()
    end = request.args.get("to", "").strip()

    single = request.args.get("date", "").strip()

    if single and not start and not end:
        start = end = single

    kind = request.args.get("type", "").strip()

    if kind not in ("", "DROWSY", "YAWNING"):
        abort(
            400,
            description="Loai canh bao khong hop le"
        )

    try:
        d1 = date.fromisoformat(start) if start else None
        d2 = date.fromisoformat(end) if end else None

    except ValueError:
        abort(
            400,
            description="Ngay phai co dang YYYY-MM-DD"
        )

    if d1 and d2 and d1 > d2:
        abort(
            400,
            description="Ngay bat dau sau ngay ket thuc"
        )

    where = []
    params = []

    if d1:
        where.append("date(created_at)>=?")
        params.append(d1.isoformat())

    if d2:
        where.append("date(created_at)<=?")
        params.append(d2.isoformat())

    if kind:
        where.append("event_type=?")
        params.append(kind)

    where_sql = (
        " WHERE " + " AND ".join(where)
        if where else ""
    )

    return where_sql, params, d1, d2, kind


# =====================================
# FONT TIENG VIET
# =====================================

def _pdf_font():
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/tahoma.ttf"),
        Path(
            "/usr/share/fonts/truetype/dejavu/"
            "DejaVuSans.ttf"
        ),
    ]

    for path in candidates:
        if path.is_file():
            name = "DriveGuardUnicode"

            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(
                    TTFont(name, str(path))
                )

            return name

    raise RuntimeError(
        "Khong tim thay font Unicode de tao PDF tieng Viet"
    )


def _fmt_number(number):
    return "-" if number is None else f"{number:.3f}"


# =====================================
# FOOTER PDF
# =====================================

def _footer(canvas, doc):
    from reportlab.lib.units import mm

    canvas.saveState()

    canvas.setFont("Helvetica", 8)
    canvas.setFillColorRGB(0.42, 0.48, 0.54)

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


# =====================================
# XUAT BAO CAO PDF
# =====================================

@bp.get("/api/export/pdf")
def export_pdf():
    _local_only()

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm

    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image,
        PageBreak
    )

    from reportlab.graphics.shapes import (
        Drawing,
        Rect,
        String
    )

    from reportlab.lib.utils import ImageReader

    where, params, d1, d2, kind = _parse_filters()

    with _connect() as db:
        rows = db.execute(
            """
            SELECT
                id,
                event_type,
                ear,
                mar,
                created_at,
                evidence_path
            FROM alerts
            """
            + where +
            " ORDER BY id DESC",
            params
        ).fetchall()

    if len(rows) > MAX_ROWS_PDF:
        abort(
            413,
            description="Qua nhieu ban ghi, hay loc theo ngay"
        )

    # Mau sac bao cao
    font = _pdf_font()

    navy = colors.HexColor("#0B1E32")
    teal = colors.HexColor("#12BBD0")
    light = colors.HexColor("#E5F5F8")

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
        fontSize=18,
        leading=24,
        spaceAfter=9
    )

    h2 = ParagraphStyle(
        "DG_H2",
        parent=normal,
        fontSize=12,
        leading=17,
        spaceBefore=9,
        spaceAfter=8
    )

    small = ParagraphStyle(
        "DG_Small",
        parent=normal,
        fontSize=8,
        leading=12
    )

    center = ParagraphStyle(
        "DG_Center",
        parent=small,
        alignment=1
    )

    # Tao file PDF trong bo nho
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

    # Thong ke
    count = len(rows)

    drowsy = sum(
        r["event_type"] == "DROWSY"
        for r in rows
    )

    yawning = sum(
        r["event_type"] == "YAWNING"
        for r in rows
    )

    rate = (
        drowsy / count * 100
        if count else 0
    )

    # =================================
    # TRANG TONG QUAN
    # =================================

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
        f"{d1.isoformat() if d1 else 'Tất cả'}"
        f" đến "
        f"{d2.isoformat() if d2 else 'Tất cả'}"
    )

    desc = {
        "": "Tất cả",
        "DROWSY": "Buồn ngủ",
        "YAWNING": "Ngáp"
    }[kind]

    story.append(
        Paragraph(
            f"Phạm vi: {period} | Loại: {desc}",
            normal
        )
    )

    story.append(Spacer(1, 6 * mm))

    # =================================
    # 4 THE THONG KE
    # =================================

    summary = Table(
        [
            [
                "TỔNG CẢNH BÁO",
                "BUỒN NGỦ",
                "NGÁP",
                "TỶ LỆ BUỒN NGỦ"
            ],
            [
                str(count),
                str(drowsy),
                str(yawning),
                f"{rate:.1f}%"
            ]
        ],
        colWidths=[43.5 * mm] * 4,
        rowHeights=[11 * mm, 15 * mm]
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
                "TEXTCOLOR",
                (0, 1),
                (-1, 1),
                navy
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
    story.append(Spacer(1, 5 * mm))

    # =================================
    # BIEU DO CANH BAO
    # =================================

    story.append(
        Paragraph(
            "PHÂN BỐ CẢNH BÁO",
            h2
        )
    )

    chart = Drawing(470, 72)

    max_count = max(
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

    for idx, (label, value, bar_color) in enumerate(
        chart_data
    ):
        y = 42 - idx * 34

        chart.add(
            String(
                0,
                y + 5,
                label,
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
                290 * value / max_count,
                14,
                fillColor=bar_color,
                strokeColor=None
            )
        )

        chart.add(
            String(
                406,
                y + 4,
                str(value),
                fontName=font,
                fontSize=10
            )
        )

    story.append(chart)

    story.append(
        Paragraph(
            "Lưu ý: Đây là số sự kiện do phần mềm ghi nhận, "
            "không phải độ chính xác hay số lần phát hiện "
            "đúng đã được xác minh.",
            small
        )
    )

    story.append(Spacer(1, 5 * mm))

    # =================================
    # BANG LICH SU CANH BAO
    # =================================

    story.append(
        Paragraph(
            "CHI TIẾT LỊCH SỬ CẢNH BÁO",
            h2
        )
    )

    table_rows = [
        [
            "ID",
            "Loại",
            "EAR",
            "MAR",
            "Thời điểm",
            "Ảnh"
        ]
    ]

    for r in rows:
        path = _image_path(
            r["id"],
            r["evidence_path"]
        )

        has_photo = bool(
            path and path.is_file()
        )

        table_rows.append([
            str(r["id"]),
            (
                "Buồn ngủ"
                if r["event_type"] == "DROWSY"
                else "Ngáp"
            ),
            _fmt_number(r["ear"]),
            _fmt_number(r["mar"]),
            str(r["created_at"]).replace(
                "T",
                " "
            ),
            "Có" if has_photo else "-"
        ])

    table = Table(
        table_rows,
        colWidths=[
            14 * mm,
            31 * mm,
            19 * mm,
            19 * mm,
            70 * mm,
            21 * mm
        ],
        repeatRows=1,
        hAlign="LEFT"
    )

    table.setStyle(
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
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                navy
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
                colors.HexColor("#D9E1E7")
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

    story.append(table)

    # =================================
    # ANH MINH CHUNG
    # =================================

    if request.args.get("images") == "1":

        selected = []

        for r in rows:
            path = _image_path(
                r["id"],
                r["evidence_path"]
            )

            if path and path.is_file():
                selected.append((r, path))

            if len(selected) == MAX_IMAGES_PDF:
                break

        if selected:

            story.append(PageBreak())

            story.append(
                Paragraph(
                    "ẢNH MINH CHỨNG CẢNH BÁO",
                    title
                )
            )

            story.append(
                Paragraph(
                    "Ảnh chỉ xuất hiện nếu chức năng lưu ảnh "
                    "được bật trước khi ghi nhận sự kiện. "
                    "Tối đa 12 ảnh gần nhất.",
                    small
                )
            )

            story.append(Spacer(1, 4 * mm))

            # Moi trang 4 anh, 2 cot x 2 hang
            for start_index in range(
                0,
                len(selected),
                4
            ):

                if start_index:
                    story.append(PageBreak())

                    story.append(
                        Paragraph(
                            "ẢNH MINH CHỨNG (TIẾP)",
                            h2
                        )
                    )

                cells = []

                for r, path in selected[
                    start_index:start_index + 4
                ]:

                    img_w, img_h = ImageReader(
                        str(path)
                    ).getSize()

                    max_w = 75 * mm
                    max_h = 52 * mm

                    factor = min(
                        max_w / img_w,
                        max_h / img_h
                    )

                    img = Image(
                        str(path),
                        width=img_w * factor,
                        height=img_h * factor
                    )

                    label = (
                        "Buồn ngủ"
                        if r["event_type"] == "DROWSY"
                        else "Ngáp"
                    )

                    caption = Paragraph(
                        f"<b>#{r['id']} - {label}</b><br/>"
                        f"{escape(str(r['created_at']).replace('T', ' '))}<br/>"
                        f"EAR: {_fmt_number(r['ear'])} | "
                        f"MAR: {_fmt_number(r['mar'])}",
                        center
                    )

                    cells.append([
                        img,
                        Spacer(1, 2 * mm),
                        caption
                    ])

                # Du 4 o cho luoi 2x2
                while len(cells) < 4:
                    cells.append("")

                gallery = Table(
                    [
                        cells[0:2],
                        cells[2:4]
                    ],
                    colWidths=[
                        87 * mm,
                        87 * mm
                    ],
                    rowHeights=[
                        93 * mm,
                        93 * mm
                    ]
                )

                gallery.setStyle(
                    TableStyle([
                        (
                            "BOX",
                            (0, 0),
                            (-1, -1),
                            0.4,
                            colors.HexColor("#D9E1E7")
                        ),
                        (
                            "INNERGRID",
                            (0, 0),
                            (-1, -1),
                            0.4,
                            colors.HexColor("#D9E1E7")
                        ),
                        (
                            "VALIGN",
                            (0, 0),
                            (-1, -1),
                            "MIDDLE"
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
                        )
                    ])
                )

                story.append(gallery)

    # =================================
    # TAO FILE PDF
    # =================================

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
# API XEM ANH MINH CHUNG
# =====================================

@bp.get("/api/evidence/<int:alert_id>")
def get_evidence(alert_id):
    _local_only()

    with _connect() as db:
        row = db.execute(
            """
            SELECT evidence_path
            FROM alerts
            WHERE id = ?
            """,
            (alert_id,)
        ).fetchone()

    if not row:
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

    response.headers["Cache-Control"] = "no-store"

    return response


# =====================================
# API XOA ANH MINH CHUNG
# =====================================

@bp.delete("/api/evidence/<int:alert_id>")
def delete_evidence(alert_id):
    _local_only()

    with _connect() as db:
        row = db.execute(
            """
            SELECT evidence_path
            FROM alerts
            WHERE id = ?
            """,
            (alert_id,)
        ).fetchone()

        if row is None:
            abort(404)

        path = _image_path(
            alert_id,
            row["evidence_path"]
        )

        db.execute(
            """
            UPDATE alerts
            SET evidence_path = NULL
            WHERE id = ?
            """,
            (alert_id,)
        )

    if path:
        path.unlink(missing_ok=True)

    return jsonify({
        "success": True,
        "alert_id": alert_id
    })


# =====================================
# DANG KY MODULE FLASK
# =====================================

def install(app):
    upgrade_schema()
    app.register_blueprint(bp)
