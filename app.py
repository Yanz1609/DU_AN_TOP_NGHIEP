from location_bridge import install as install_location_bridge
from flask import Flask, render_template, Response, jsonify, request
from camera_stream import CameraStream
import sqlite3
import csv
import json
import time
import io
from datetime import date

from evidence_reports import install as install_evidence_reports
from location_bridge import (
    install as install_driver_location,
    clear_location
)

app = Flask(__name__)

camera_stream = CameraStream()

# Dang ky module PDF va anh minh chung
install_evidence_reports(app)

# Dang ky module GPS
install_location_bridge(app, camera_stream)

# Dang ky GPS Bridge sau khi khoi tao camera
install_driver_location(app, camera_stream)


# ==========================================
# CAMERA STREAM
# ==========================================

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
        clear_location()

        return jsonify({
            "success": False,
            "message": "Khong the khoi dong webcam"
        }), 503


# ==========================================
# STATISTICS
# ==========================================

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


# ==========================================
# DASHBOARD
# ==========================================

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


# ==========================================
# API STATISTICS
# ==========================================

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


# ==========================================
# API STATUS
# ==========================================

@app.route("/api/status")
def api_status():
    return jsonify(camera_stream.get_status())


# ==========================================
# SERVER-SENT EVENTS
# ==========================================

@app.route("/api/status/stream")
def api_status_stream():
    def generate():
        last_payload = None

        try:
            while True:
                data = camera_stream.get_status()

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

                if message != last_payload:
                    yield f"data: {message}\n\n"
                    last_payload = message

                time.sleep(0.05)

        except GeneratorExit:
            return

    return Response(
        generate(),
        content_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ==========================================
# STOP CAMERA + CLEAR GPS
# ==========================================

@app.route("/api/camera/stop", methods=["POST"])
def stop_camera():
    camera_stream.stop()

    # Xoa vi tri luu tam khi camera tat
    clear_location()

    return jsonify({
        "success": True,
        "message": "Camera da tat"
    })


# ==========================================
# HISTORY WITH FILTER + PAGINATION
# ==========================================

@app.route("/api/history")
def api_history():
    event_type = request.args.get("type", "")
    selected_date = request.args.get("date", "")

    page = request.args.get("page", 1, type=int) or 1
    page_size = request.args.get("page_size", 10, type=int) or 10

    page = max(1, page)

    if page_size not in (10, 20, 50):
        page_size = 10

    if event_type not in ("", "DROWSY", "YAWNING"):
        return jsonify({
            "error": "Loai canh bao khong hop le"
        }), 400

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

        count_sql = (
            "SELECT COUNT(*) AS total FROM alerts"
            + where_sql
        )

        total = conn.execute(
            count_sql,
            params
        ).fetchone()["total"]

        total_pages = max(
            1,
            (total + page_size - 1) // page_size
        )

        page = min(page, total_pages)

        offset = (page - 1) * page_size

        query = """
            SELECT id, event_type, ear, mar, created_at
            FROM alerts
        """ + where_sql + """
            ORDER BY id DESC
            LIMIT ? OFFSET ?
        """

        rows = conn.execute(
            query,
            params + [page_size, offset]
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


# ==========================================
# EXPORT CSV
# ==========================================

@app.route("/api/export/csv")
def export_csv():
    event_type = request.args.get("type", "")
    selected_date = request.args.get("date", "")

    if event_type not in ("", "DROWSY", "YAWNING"):
        return jsonify({
            "error": "Loai canh bao khong hop le"
        }), 400

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

        rows = conn.execute(
            query,
            params
        ).fetchall()

    finally:
        conn.close()

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

    csv_content = "\ufeff" + output.getvalue()

    return Response(
        csv_content,
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition":
                'attachment; filename="lich_su_canh_bao.csv"'
        }
    )


# ==========================================
# RUN FLASK
# ==========================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
