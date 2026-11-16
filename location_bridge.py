
"""DriveGuard AI - Ket noi vi tri laptop voi Flask."""

import math
import threading
import time
from datetime import datetime

from flask import Blueprint, abort, jsonify, request

bp_location = Blueprint(
    "driver_location_bridge",
    __name__
)

_lock = threading.Lock()
_latest = None
_camera = None

# Chi dung vi tri duoc cap nhat trong 30 giay
MAX_AGE_SECONDS = 30


# ====================================
# BAO VE API LOCALHOST
# ====================================

def _only_local():
    if request.remote_addr not in ("127.0.0.1", "::1"):
        abort(403)

    origin = request.headers.get("Origin")

    if origin and origin.rstrip("/") != request.host_url.rstrip("/"):
        abort(403)


# ====================================
# KIEM TRA CAMERA
# ====================================

def _camera_on():
    return (
        _camera is not None
        and bool(getattr(_camera, "running", False))
    )


# ====================================
# LAY VI TRI MOI NHAT
# ====================================

def get_recent_location():
    with _lock:
        result = dict(_latest) if _latest else None

    if not result or not _camera_on():
        return None

    age = time.monotonic() - result["_received_monotonic"]

    if age > MAX_AGE_SECONDS:
        return None

    result.pop("_received_monotonic", None)

    return result


# ====================================
# XOA VI TRI HIEN TAI
# ====================================

def clear_location():
    global _latest

    with _lock:
        _latest = None


# ====================================
# API NHAN VI TRI TU TRINH DUYET
# ====================================

@bp_location.post("/api/location/update")
def update_location():
    global _latest

    _only_local()

    if not _camera_on():
        clear_location()

        return jsonify({
            "ok": False,
            "error": "camera_off"
        }), 409

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "ok": False,
            "error": "invalid_json"
        }), 400

    try:
        latitude = float(data["latitude"])
        longitude = float(data["longitude"])
        accuracy = float(data["accuracy"])

    except (KeyError, ValueError, TypeError, OverflowError):
        return jsonify({
            "ok": False,
            "error": "invalid_coordinates"
        }), 400

    if not all(
        math.isfinite(value)
        for value in (latitude, longitude, accuracy)
    ):
        return jsonify({
            "ok": False,
            "error": "invalid_coordinates"
        }), 400

    if not (
        -90 <= latitude <= 90
        and -180 <= longitude <= 180
        and 0 <= accuracy <= 100000
    ):
        return jsonify({
            "ok": False,
            "error": "invalid_coordinates"
        }), 400

    now = datetime.now().astimezone()

    location = {
        "latitude": latitude,
        "longitude": longitude,
        "accuracy": accuracy,
        "recorded_at": now.isoformat(
            timespec="seconds"
        ),
        "_received_monotonic": time.monotonic()
    }

    with _lock:
        _latest = location

    return jsonify({
        "ok": True
    })


# ====================================
# API DUNG DINH VI
# ====================================

@bp_location.post("/api/location/clear")
def stop_location():
    _only_local()

    clear_location()

    return jsonify({
        "ok": True
    })


# ====================================
# DANG KY BLUEPRINT
# ====================================

def install(app, camera_stream):
    global _camera

    _camera = camera_stream

    if "driver_location_bridge" not in app.blueprints:
        app.register_blueprint(bp_location)
