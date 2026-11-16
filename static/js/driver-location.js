
/*
 * DRIVEGUARD AI - OPTIMIZED LAPTOP LOCATION
 * Windows 11 + Chrome + Flask
 * Khong can dien thoai hoac GPS USB.
 */

(() => {
    "use strict";

    // =====================================
    // CAU HINH
    // =====================================

    const CONFIG = {
        checkInterval: 12000,
        highAccuracyTimeout: 20000,
        fallbackTimeout: 15000,
        maxFixAge: 25000,
        maxAccuracy: 5000
    };

    let map = null;
    let marker = null;
    let accuracyCircle = null;

    let tracking = false;
    let watchId = null;
    let checkTimer = null;

    let sessionId = 0;
    let requestPending = false;
    let sending = false;

    let latestPosition = null;
    let lastSentTimestamp = 0;
    let lastServerSuccess = 0;
    let queuedPosition = null;

    const el = id => document.getElementById(id);

    function write(id, value) {
        const target = el(id);

        if (target) {
            target.textContent = value;
        }
    }

    // =====================================
    // TRANG THAI GPS
    // =====================================

    function setStatus(message, state = "") {
        const badge = el("driverGpsStatus");

        if (!badge) return;

        badge.textContent = message;
        badge.className = "dg-gps-status";

        if (state) {
            badge.classList.add(state);
        }
    }

    // =====================================
    // KHOI TAO BAN DO
    // =====================================

    function initMap() {
        if (map) return true;

        if (!el("driverMap") || !window.L) {
            return false;
        }

        map = window.L.map("driverMap", {
            scrollWheelZoom: false
        }).setView([16.05, 108.20], 12);

        window.L.tileLayer(
            "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            {
                maxZoom: 19,
                attribution: "&copy; OpenStreetMap contributors"
            }
        ).addTo(map);

        setTimeout(() => {
            map?.invalidateSize();
        }, 250);

        return true;
    }

    // =====================================
    // KIEM TRA TOA DO
    // =====================================

    function parsePosition(position) {
        if (!position || !position.coords) {
            return null;
        }

        const latitude = Number(position.coords.latitude);
        const longitude = Number(position.coords.longitude);
        const accuracy = Number(position.coords.accuracy);
        const timestamp = Number(position.timestamp);

        if (
            ![
                latitude,
                longitude,
                accuracy,
                timestamp
            ].every(Number.isFinite)
        ) {
            return null;
        }

        if (
            latitude < -90 ||
            latitude > 90 ||
            longitude < -180 ||
            longitude > 180 ||
            accuracy < 0 ||
            accuracy > CONFIG.maxAccuracy
        ) {
            return null;
        }

        const age = Date.now() - timestamp;

        if (
            age < -5000 ||
            age > CONFIG.maxFixAge
        ) {
            return null;
        }

        return {
            latitude,
            longitude,
            accuracy,
            timestamp
        };
    }

    function isFresh(position) {
        if (!position) return false;

        const age = Date.now() - position.timestamp;

        return age >= 0 && age <= CONFIG.maxFixAge;
    }

    // =====================================
    // HIEN THI VI TRI
    // =====================================

    function renderPosition(fix) {
        write(
            "driverLatitude",
            fix.latitude.toFixed(6)
        );

        write(
            "driverLongitude",
            fix.longitude.toFixed(6)
        );

        write(
            "driverAccuracy",
            `±${fix.accuracy.toFixed(1)} m`
        );

        write(
            "driverLocationTime",
            new Date(fix.timestamp).toLocaleTimeString(
                "vi-VN"
            )
        );

        const mapButton = el("openDriverMap");

        if (mapButton) {
            mapButton.disabled = false;
        }

        if (!initMap()) return;

        const point = [
            fix.latitude,
            fix.longitude
        ];

        if (!marker) {
            marker = window.L.marker(point)
                .addTo(map)
                .bindPopup(
                    "Vị trí ước tính của laptop"
                );

            map.setView(point, 16);
        } else {
            marker.setLatLng(point);
        }

        if (!accuracyCircle) {
            accuracyCircle = window.L.circle(
                point,
                {
                    radius: fix.accuracy,
                    color: "#10cde1",
                    fillColor: "#10cde1",
                    fillOpacity: 0.12,
                    weight: 2
                }
            ).addTo(map);
        } else {
            accuracyCircle.setLatLng(point);
            accuracyCircle.setRadius(fix.accuracy);
        }
    }

    // =====================================
    // DONG BO TOA DO VOI FLASK
    // =====================================

    async function syncLocation(fix, currentSession) {
        queuedPosition = fix;

        if (sending) return;

        sending = true;

        try {
            while (
                tracking &&
                sessionId === currentSession &&
                queuedPosition
            ) {
                const current = queuedPosition;
                queuedPosition = null;

                if (
                    !isFresh(current) ||
                    current.timestamp <= lastSentTimestamp
                ) {
                    continue;
                }

                let response;

                try {
                    response = await fetch(
                        "/api/location/update",
                        {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json"
                            },
                            cache: "no-store",
                            body: JSON.stringify({
                                latitude: current.latitude,
                                longitude: current.longitude,
                                accuracy: current.accuracy
                            })
                        }
                    );
                } catch (error) {
                    if (
                        tracking &&
                        sessionId === currentSession
                    ) {
                        console.warn(
                            "[GPS] Khong ket noi duoc Flask:",
                            error
                        );

                        setStatus(
                            "LỖI KẾT NỐI GPS",
                            "error"
                        );
                    }

                    continue;
                }

                if (
                    !tracking ||
                    sessionId !== currentSession
                ) {
                    break;
                }

                if (!response.ok) {
                    console.warn(
                        "[GPS] Backend tra ve:",
                        response.status
                    );

                    setStatus(
                        `LỖI GPS HTTP ${response.status}`,
                        "error"
                    );

                    continue;
                }

                let result;

                try {
                    result = await response.json();
                } catch {
                    continue;
                }

                if (result.ok !== true) {
                    setStatus(
                        "FLASK CHƯA NHẬN GPS",
                        "error"
                    );

                    continue;
                }

                lastSentTimestamp = current.timestamp;
                lastServerSuccess = Date.now();

                setStatus(
                    "ĐANG ĐỊNH VỊ",
                    "active"
                );

                console.log(
                    "[GPS] Flask da nhan toa do:",
                    current.latitude,
                    current.longitude,
                    "Sai so:",
                    current.accuracy.toFixed(1),
                    "m"
                );
            }
        } finally {
            sending = false;
        }
    }

    // =====================================
    // NHAN VI TRI TU CHROME
    // =====================================

    function onPosition(position, currentSession) {
        if (
            !tracking ||
            sessionId !== currentSession
        ) {
            return;
        }

        const fix = parsePosition(position);

        if (!fix) {
            console.warn(
                "[GPS] Vi tri khong hop le hoac qua cu"
            );

            return;
        }

        // Khong cho du lieu cu ghi de du lieu moi
        if (
            latestPosition &&
            fix.timestamp < latestPosition.timestamp
        ) {
            return;
        }

        // Neu hai ket qua cung thoi diem,
        // chon ket qua co sai so nho hon.
        if (
            latestPosition &&
            fix.timestamp === latestPosition.timestamp &&
            fix.accuracy >= latestPosition.accuracy
        ) {
            return;
        }

        latestPosition = fix;

        renderPosition(fix);

        void syncLocation(
            fix,
            currentSession
        );
    }

    // =====================================
    // XU LY LOI GPS
    // =====================================

    function onError(error, currentSession) {
        if (
            !tracking ||
            sessionId !== currentSession
        ) {
            return;
        }

        console.warn(
            "[GPS] Error:",
            error.code,
            error.message
        );

        if (error.code === 1) {
            stopDriverLocation();

            setStatus(
                "CHƯA CẤP QUYỀN VỊ TRÍ",
                "error"
            );

            return;
        }

        const hasRecentLocation =
            isFresh(latestPosition) &&
            lastServerSuccess > 0 &&
            Date.now() - lastServerSuccess <
                CONFIG.maxFixAge;

        if (hasRecentLocation) {
            setStatus(
                "ĐANG ĐỊNH VỊ",
                "active"
            );
        } else {
            setStatus(
                "GPS ĐANG THỬ LẠI",
                "error"
            );
        }

        // Timeout KHONG lam dung GPS.
    }

    // =====================================
    // YEU CAU VI TRI MOI
    // =====================================

    function requestPosition(currentSession) {
        if (
            !tracking ||
            sessionId !== currentSession ||
            requestPending
        ) {
            return;
        }

        requestPending = true;

        // Uu tien do chinh xac cao truoc.
        navigator.geolocation.getCurrentPosition(
            position => {
                requestPending = false;

                onPosition(
                    position,
                    currentSession
                );
            },
            error => {
                if (
                    !tracking ||
                    sessionId !== currentSession
                ) {
                    requestPending = false;
                    return;
                }

                if (error.code === 1) {
                    requestPending = false;

                    onError(
                        error,
                        currentSession
                    );

                    return;
                }

                // Neu high accuracy bi timeout,
                // thu nguon vi tri du phong.
                navigator.geolocation.getCurrentPosition(
                    position => {
                        requestPending = false;

                        onPosition(
                            position,
                            currentSession
                        );
                    },
                    fallbackError => {
                        requestPending = false;

                        onError(
                            fallbackError,
                            currentSession
                        );
                    },
                    {
                        enableHighAccuracy: false,
                        maximumAge: 0,
                        timeout: CONFIG.fallbackTimeout
                    }
                );
            },
            {
                enableHighAccuracy: true,
                maximumAge: 0,
                timeout: CONFIG.highAccuracyTimeout
            }
        );
    }

    // =====================================
    // START GPS
    // =====================================

    function startDriverLocation() {
        if (tracking) return;

        if (!navigator.geolocation) {
            setStatus(
                "TRÌNH DUYỆT KHÔNG HỖ TRỢ GPS",
                "error"
            );
            return;
        }

        if (!window.isSecureContext) {
            setStatus(
                "CẦN LOCALHOST HOẶC HTTPS",
                "error"
            );
            return;
        }

        tracking = true;
        sessionId++;

        const currentSession = sessionId;

        requestPending = false;
        latestPosition = null;
        queuedPosition = null;
        lastSentTimestamp = 0;
        lastServerSuccess = 0;

        initMap();

        setStatus("ĐANG XÁC ĐỊNH VỊ TRÍ");

        // Theo doi vi tri lien tuc.
        watchId = navigator.geolocation.watchPosition(
            position => {
                onPosition(
                    position,
                    currentSession
                );
            },
            error => {
                onError(
                    error,
                    currentSession
                );
            },
            {
                enableHighAccuracy: true,
                maximumAge: 0,
                timeout: 30000
            }
        );

        // Lay vi tri ngay khi bat camera.
        requestPosition(currentSession);

        // Dinh ky thu lai khi laptop dung yen.
        checkTimer = window.setInterval(() => {
            if (
                !tracking ||
                sessionId !== currentSession
            ) {
                return;
            }

            requestPosition(currentSession);

            if (
                lastServerSuccess > 0 &&
                Date.now() - lastServerSuccess > 30000
            ) {
                setStatus(
                    "ĐANG CHỜ GPS MỚI",
                    "error"
                );
            }
        }, CONFIG.checkInterval);
    }

    // =====================================
    // STOP GPS
    // =====================================

    function stopDriverLocation() {
        const wasTracking = tracking;

        tracking = false;
        sessionId++;

        if (watchId !== null) {
            navigator.geolocation.clearWatch(watchId);
            watchId = null;
        }

        if (checkTimer !== null) {
            window.clearInterval(checkTimer);
            checkTimer = null;
        }

        requestPending = false;
        queuedPosition = null;
        latestPosition = null;
        lastSentTimestamp = 0;
        lastServerSuccess = 0;

        if (map && marker) {
            map.removeLayer(marker);
            marker = null;
        }

        if (map && accuracyCircle) {
            map.removeLayer(accuracyCircle);
            accuracyCircle = null;
        }

        [
            "driverLatitude",
            "driverLongitude",
            "driverAccuracy",
            "driverLocationTime"
        ].forEach(id => {
            write(id, "--");
        });

        const button = el("openDriverMap");

        if (button) {
            button.disabled = true;
        }

        setStatus("ĐÃ DỪNG ĐỊNH VỊ");

        if (wasTracking) {
            fetch(
                "/api/location/clear",
                {
                    method: "POST",
                    cache: "no-store",
                    keepalive: true
                }
            ).catch(() => {});
        }
    }

    // =====================================
    // GOOGLE MAPS
    // =====================================

    function openDriverLocation() {
        if (
            !tracking ||
            !isFresh(latestPosition)
        ) {
            return;
        }

        const fix = latestPosition;

        const url =
            "https://www.google.com/maps/search/" +
            "?api=1&query=" +
            encodeURIComponent(
                `${fix.latitude},${fix.longitude}`
            );

        window.open(
            url,
            "_blank",
            "noopener,noreferrer"
        );
    }

    // =====================================
    // PUBLIC API
    // =====================================

    window.startDriverLocation =
        startDriverLocation;

    window.stopDriverLocation =
        stopDriverLocation;

    window.openDriverLocation =
        openDriverLocation;

    window.addEventListener(
        "pagehide",
        stopDriverLocation
    );
})();
