
from flask import Flask, render_template, Response, jsonify, request
from camera_stream import CameraStream
import sqlite3
import csv
import json
import time
import io
from datetime import date
from evidence_reports import install as install_evidence_reports

app = Flask(__name__)

install_evidence_reports(app)

camera_stream = CameraStream()


@app.route("/video_feed")
def video_feed():
    try:
        camera_stream.start()

        return Response(
            camera_stream.generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    except Exception as e:
        print("LOI KHOI DONG CAMERA:", e)

        camera_stream.stop()

        return jsonify({
            "success": False,
            "message": "Khong the khoi dong webcam"
        }), 503

def get_statistics():
    conn = sqlite3.connect("drowsiness.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM alerts")
    total = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM alerts WHERE event_type = 'DROWSY'"
    )
    drowsy = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM alerts WHERE event_type = 'YAWNING'"
    )
    yawning = cursor.fetchone()[0]

    cursor.execute("""
        SELECT event_type, ear, mar, created_at
        FROM alerts
        ORDER BY id DESC
        LIMIT 20
    """)
    history = cursor.fetchall()

    conn.close()

    return total, drowsy, yawning, history


@app.route("/")
def dashboard():
    total, drowsy, yawning, history = get_statistics()

    return render_template(
        "index.html",
        total=total,
        drowsy=drowsy,
        yawning=yawning,
        history=history
    )


# API cung cap thong ke va lich su canh bao
@app.route("/api/statistics")
def api_statistics():
    total, drowsy, yawning, history = get_statistics()

    return jsonify({
        "total": total,
        "drowsy": drowsy,
        "yawning": yawning,
        "history": [
            {
                "event_type": row[0],
                "ear": row[1],
                "mar": row[2],
                "created_at": row[3]
            }
            for row in history
        ]
    })


# API lay trang thai tai xe truc tiep
@app.route("/api/status")
def api_status():
    return jsonify(camera_stream.get_status())

@app.route("/api/status/stream")
def api_status_stream():
    def generate():
        last_payload = None

        try:
            while True:
                data = camera_stream.get_status()

                # Lam tron chi so de giam du lieu khong can thiet
                payload = dict(data)

                if payload.get("ear") is not None:
                    payload["ear"] = round(
                        float(payload["ear"]), 4
                    )

                if payload.get("mar") is not None:
                    payload["mar"] = round(
                        float(payload["mar"]), 4
                    )

                message = json.dumps(
                    payload,
                    ensure_ascii=False
                )

                # Chi gui khi du lieu thuc su thay doi
                if message != last_payload:
                    yield f"data: {message}\n\n"
                    last_payload = message

                # Kiem tra trang thai moi toi da 20 lan/giay
                time.sleep(0.05)

        except GeneratorExit:
            # Trinh duyet dong ket noi SSE
            return

    return Response(
        generate(),
        content_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# API tat camera
@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    camera_stream.stop()

    return jsonify({
        "success": True,
        "message": "Camera da tat"
    })


# API lich su canh bao co bo loc

@app.route("/api/history")
def api_history():
    event_type = request.args.get("type", "")
    selected_date = request.args.get("date", "")

    # Phan trang
    page = request.args.get("page", 1, type=int) or 1
    page_size = request.args.get("page_size", 10, type=int) or 10

    page = max(1, page)

    # Chi cho phep cac kich thuoc trang hop le
    if page_size not in (10, 20, 50):
        page_size = 10

    # Kiem tra loai canh bao
    if event_type not in ("", "DROWSY", "YAWNING"):
        return jsonify({
            "error": "Loai canh bao khong hop le"
        }), 400

    # Kiem tra ngay
    if selected_date:
        try:
            selected_date = date.fromisoformat(
                selected_date
            ).isoformat()
        except ValueError:
            return jsonify({
                "error": "Ngay khong hop le"
            }), 400

    conn = sqlite3.connect("drowsiness.db")
    conn.row_factory = sqlite3.Row

    try:
        # Tao dieu kien loc dung chung cho truy van dem va truy van du lieu
        conditions = []
        params = []

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)

        if selected_date:
            conditions.append("date(created_at) = ?")
            params.append(selected_date)

        where_sql = (
            " WHERE " + " AND ".join(conditions)
            if conditions
            else ""
        )

        # Dem tong so ban ghi phu hop voi bo loc
        count_sql = (
            "SELECT COUNT(*) AS total FROM alerts"
            + where_sql
        )

        total = conn.execute(
            count_sql,
            params
        ).fetchone()["total"]

        # Tinh tong so trang
        total_pages = max(
            1,
            (total + page_size - 1) // page_size
        )

        # Neu trang vuot qua gioi han thi dua ve trang cuoi
        page = min(page, total_pages)

        offset = (page - 1) * page_size

        # Chi lay du lieu cua trang hien tai
        query = """
            SELECT id, event_type, ear, mar, created_at
            FROM alerts
        """ + where_sql + """
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """

        query_params = params + [page_size, offset]

        rows = conn.execute(
            query,
            query_params
        ).fetchall()

        history = [dict(row) for row in rows]

        return jsonify({
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "history": history
        })

    finally:
        conn.close()

# ===== XUAT LICH SU CANH BAO RA CSV =====
@app.route("/api/export/csv")
def export_csv():
    event_type = request.args.get("type", "")
    selected_date = request.args.get("date", "")

    # Kiem tra loai canh bao
    if event_type not in ("", "DROWSY", "YAWNING"):
        return jsonify({"error": "Loai canh bao khong hop le"}), 400

    # Kiem tra ngay
    if selected_date:
        try:
            selected_date = date.fromisoformat(
                selected_date
            ).isoformat()
        except ValueError:
            return jsonify({"error": "Ngay khong hop le"}), 400

    conn = sqlite3.connect("drowsiness.db")

    try:
        query = """
            SELECT id, event_type, ear, mar, created_at
            FROM alerts
            WHERE 1 = 1
        """

        params = []

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if selected_date:
            query += " AND date(created_at) = ?"
            params.append(selected_date)

        query += " ORDER BY id DESC"

        rows = conn.execute(query, params).fetchall()

    finally:
        conn.close()

    # Tao noi dung CSV trong bo nho
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "ID",
        "Loai canh bao",
        "EAR",
        "MAR",
        "Thoi gian"
    ])

    for row in rows:
        event_name = (
            "Ngu gat" if row[1] == "DROWSY"
            else "Ngap" if row[1] == "YAWNING"
            else row[1]
        )

        writer.writerow([
            row[0],
            event_name,
            row[2],
            row[3],
            row[4]
        ])

    # BOM giup Excel doc tieng Viet tot hon
    csv_content = "\ufeff" + output.getvalue()

    return Response(
        csv_content,
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition":
                'attachment; filename="lich_su_canh_bao.csv"'
        }
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )