import os
import io
import cv2
import base64
import numpy as np
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from openpyxl import Workbook

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Natijalarni vaqtinchalik xotirada saqlash
baza_natijalar = []

def nuqtalarni_tartiblash(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def markerni_topish_va_tekislash(rasm):
    kulrang = cv2.cvtColor(rasm, cv2.COLOR_BGR2GRAY)
    xira = cv2.GaussianBlur(kulrang, (5, 5), 0)
    binar = cv2.adaptiveThreshold(xira, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 10)

    konturlar, _ = cv2.findContours(binar, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    markerlar = []
    for k in konturlar:
        perimetr = cv2.arcLength(k, True)
        taqribiy = cv2.approxPolyDP(k, 0.04 * perimetr, True)
        if len(taqribiy) == 4:
            maydon = cv2.contourArea(k)
            x, y, w, h = cv2.boundingRect(taqribiy)
            nisbat = w / float(h)
            if 0.7 <= nisbat <= 1.3 and 100 < maydon < (rasm.shape[0] * rasm.shape[1] * 0.08):
                M = cv2.moments(k)
                if M["m00"] != 0:
                    markerlar.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    if len(markerlar) >= 4:
        pts = np.array(markerlar, dtype="float32")
        rect = nuqtalarni_tartiblash(pts)

        kenglik, balandlik = 600, 850
        dst = np.array([
            [53, 53],
            [kenglik - 53, 53],
            [kenglik - 53, balandlik - 53],
            [53, balandlik - 53]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(rasm, M, (kenglik, balandlik))
    return None

def javoblarni_aniqlash(tekis_rasm, kalitlar):
    kulrang = cv2.cvtColor(tekis_rasm, cv2.COLOR_BGR2GRAY)
    _, binar = cv2.threshold(kulrang, 140, 255, cv2.THRESH_BINARY_INV)

    variantlar = ['A', 'B', 'C', 'D']
    radius = 9
    gap_x = 26.2
    start_y = 202
    qator_orasi = 32.3

    talaba_javoblari = {}
    togri_soni = 0
    natija_rasm = tekis_rasm.copy()

    for savol_raqam in range(1, 26):
        if savol_raqam <= 13:
            i = savol_raqam - 1
            y = int(start_y + (i * qator_orasi))
            start_x = 156
        else:
            i = savol_raqam - 14
            y = int(start_y + (i * qator_orasi))
            start_x = 378

        piksellar = []
        koordinatalar = []

        for idx in range(4):
            cx = int(start_x + (idx * gap_x))
            koordinatalar.append((cx, y))

            mask = np.zeros(binar.shape, dtype="uint8")
            cv2.circle(mask, (cx, y), radius - 2, 255, -1)
            qora = cv2.countNonZero(cv2.bitwise_and(binar, binar, mask=mask))
            piksellar.append(qora)

        maks_piksel = max(piksellar)
        ikkinchi_maks = sorted(piksellar)[-2]

        if maks_piksel > 35 and (maks_piksel - ikkinchi_maks) > 15:
            tanlangan_idx = int(np.argmax(piksellar))
            belgilangan = variantlar[tanlangan_idx]
        else:
            belgilangan = "Belgilanmagan"

        talaba_javoblari[savol_raqam] = belgilangan

        togri_javob = kalitlar.get(savol_raqam)
        if togri_javob:
            if belgilangan == togri_javob:
                togri_soni += 1
                cx, cy = koordinatalar[variantlar.index(togri_javob)]
                cv2.circle(natija_rasm, (cx, cy), radius + 4, (0, 200, 0), 2)
            elif belgilangan in variantlar:
                cx, cy = koordinatalar[variantlar.index(belgilangan)]
                cv2.circle(natija_rasm, (cx, cy), radius + 4, (0, 0, 255), 2)
                tcx, tcy = koordinatalar[variantlar.index(togri_javob)]
                cv2.circle(natija_rasm, (tcx, tcy), radius + 4, (0, 200, 0), 2)

    return talaba_javoblari, togri_soni, natija_rasm

def kalitlarni_olish():
    standart = "BCAD" + "A" * 21
    if not os.path.exists("kalitlar.txt"):
        with open("kalitlar.txt", "w") as f:
            f.write(standart)
        return {i + 1: standart[i] for i in range(25)}

    with open("kalitlar.txt", "r") as f:
        qator = f.read().strip().replace(" ", "").upper()
    return {i + 1: qator[i] for i in range(min(len(qator), 25))}

@app.get("/", response_class=HTMLResponse)
def index():
    kalit_matn = "".join([v for _, v in sorted(kalitlarni_olish().items())])
    return f"""
    <!DOCTYPE html>
    <html lang="uz">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OMR Test Skaner Pro</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{ font-family: system-ui, -apple-system, sans-serif; background: #f0f2f5; margin: 0; padding: 15px; color: #1f2937; }}
            .container {{ max-width: 480px; margin: auto; }}
            .card {{ background: white; padding: 20px; border-radius: 16px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 15px; }}
            h2, h3 {{ margin: 0 0 10px 0; text-align: center; }}
            .subtitle {{ color: #6b7280; font-size: 13px; text-align: center; margin-bottom: 15px; }}
            
            input[type="text"] {{ width: 100%; padding: 12px; border: 1.5px solid #d1d5db; border-radius: 8px; font-size: 14px; margin-bottom: 12px; font-weight: 500; }}
            .btn {{ display: block; width: 100%; background: #2563eb; color: white; padding: 12px; border-radius: 8px; font-size: 15px; font-weight: bold; cursor: pointer; border: none; text-align: center; }}
            .btn-green {{ background: #10b981; margin-top: 10px; }}
            .btn-outline {{ background: white; color: #374151; border: 1.5px solid #d1d5db; margin-top: 10px; }}
            input[type="file"] {{ display: none; }}
            
            #loading {{ display: none; margin: 15px 0; font-weight: bold; color: #2563eb; text-align: center; }}
            .stat {{ display: flex; justify-content: space-between; background: #eff6ff; padding: 12px 16px; border-radius: 10px; font-weight: bold; font-size: 16px; margin: 15px 0; }}
            .ans-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px; max-height: 200px; overflow-y: auto; padding-right: 4px; }}
            .ans-item {{ padding: 6px 10px; border-radius: 6px; font-size: 12px; display: flex; justify-content: space-between; background: #f9fafb; border: 1px solid #e5e7eb; }}
            .correct {{ color: #16a34a; font-weight: bold; }}
            .wrong {{ color: #dc2626; font-weight: bold; }}
            
            #previewImg {{ width: 100%; border-radius: 8px; margin-top: 12px; border: 1px solid #e5e7eb; display: none; }}
            
            .table-wrap {{ max-height: 180px; overflow-y: auto; margin-top: 10px; }}
            table {{ width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; }}
            th, td {{ padding: 8px; border-bottom: 1px solid #e5e7eb; }}
            th {{ background: #f9fafb; }}
        </style>
    </head>
    <body>
        <div class="container">
            <!-- Asosiy Skaner Kartasi -->
            <div class="card">
                <h2>OMR Skaner</h2>
                <div class="subtitle">25 talik testlarni tezkor tekshirish</div>

                <input type="text" id="studentName" placeholder="O'quvchi F.I.SH yoki ID (ixtiyoriy)">

                <label class="btn" for="fileInput">Rasmga olish / Yuklash</label>
                <input type="file" id="fileInput" accept="image/*" capture="environment">

                <div id="loading">Varaq tekshirilmoqda...</div>

                <div id="result"></div>
                <img id="previewImg" src="" alt="Tekshirilgan varaq">
            </div>

            <!-- Kalitlarni sozlash kartasi -->
            <div class="card">
                <h3>To'g'ri javoblar kaliti</h3>
                <input type="text" id="keysInput" value="{kalit_matn}" maxlength="25" placeholder="25 ta javob (masalan: BCAD...)">
                <button class="btn btn-green" onclick="saveKeys()">Kalitlarni saqlash</button>
            </div>

            <!-- Natijalar ro'yxati va Excel kartasi -->
            <div class="card">
                <h3>Tekshirilganlar ro'yxati</h3>
                <div class="table-wrap">
                    <table id="historyTable">
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>F.I.SH</th>
                                <th>Ball</th>
                                <th>Foiz</th>
                            </tr>
                        </thead>
                        <tbody id="historyBody">
                            <tr><td colspan="4" style="text-align:center; color:#9ca3af;">Hozircha natijalar yo'q</td></tr>
                        </tbody>
                    </table>
                </div>
                <a href="/download-excel" class="btn btn-outline" style="text-decoration: none;">Excelda yuklab olish (.xlsx)</a>
            </div>
        </div>

        <script>
            const input = document.getElementById('fileInput');
            const studentInput = document.getElementById('studentName');
            const loading = document.getElementById('loading');
            const resultDiv = document.getElementById('result');
            const previewImg = document.getElementById('previewImg');
            const historyBody = document.getElementById('historyBody');

            input.addEventListener('change', async () => {{
                if (!input.files[0]) return;
                
                loading.style.display = 'block';
                resultDiv.innerHTML = '';
                previewImg.style.display = 'none';

                const formData = new FormData();
                formData.append('file', input.files[0]);
                formData.append('student', studentInput.value.trim() || "Noma'lum");

                try {{
                    const res = await fetch('/tekshirish', {{ method: 'POST', body: formData }});
                    const data = await res.json();
                    loading.style.display = 'none';

                    if (!data.success) {{
                        resultDiv.innerHTML = `<p style="color:red; text-align:center; font-weight:bold;">${{data.error}}</p>`;
                        return;
                    }}

                    let gridHtml = '';
                    for (const [q, val] of Object.entries(data.javoblar)) {{
                        const statusClass = val.holat === "To'g'ri" ? "correct" : (val.belgilangan === "Belgilanmagan" ? "" : "wrong");
                        gridHtml += `
                            <div class="ans-item">
                                <span>${{q}}-savol: <b>${{val.belgilangan}}</b></span>
                                <span class="${{statusClass}}">${{val.holat}}</span>
                            </div>
                        `;
                    }}

                    resultDiv.innerHTML = `
                        <div class="stat">
                            <span>${{data.student}}</span>
                            <span>${{data.ball}} / 25 (${{data.foiz}}%)</span>
                        </div>
                        <div class="ans-grid">${{gridHtml}}</div>
                    `;

                    if (data.image_base64) {{
                        previewImg.src = "data:image/jpeg;base64," + data.image_base64;
                        previewImg.style.display = 'block';
                    }}

                    updateTable(data.tarix);
                    studentInput.value = ''; // Keyingi o'quvchi uchun tozalash
                }} catch (err) {{
                    loading.style.display = 'none';
                    resultDiv.innerHTML = `<p style="color:red; text-align:center;">Serverga ulanishda xatolik!</p>`;
                }}
            }});

            async function saveKeys() {{
                const keys = document.getElementById('keysInput').value.trim();
                if (keys.length !== 25) {{
                    alert("Kalitlar soni roppa-rosa 25 ta bo'lishi kerak!");
                    return;
                }}
                const res = await fetch('/save-keys', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ keys: keys }})
                }});
                const data = await res.json();
                if (data.success) alert("Kalitlar muvaffaqiyatli saqlandi!");
            }}

            function updateTable(tarix) {{
                if (!tarix || tarix.length === 0) return;
                let rows = '';
                tarix.slice().reverse().forEach((item, idx) => {{
                    rows += `
                        <tr>
                            <td>${{tarix.length - idx}}</td>
                            <td><b>${{item.student}}</b></td>
                            <td>${{item.ball}}</td>
                            <td>${{item.foiz}}%</td>
                        </tr>
                    `;
                }});
                historyBody.innerHTML = rows;
            }}
        </script>
    </body>
    </html>
    """

@app.post("/save-keys")
async def save_keys(data: dict):
    kalitlar = data.get("keys", "").strip().upper()
    if len(kalitlar) == 25:
        with open("kalitlar.txt", "w") as f:
            f.write(kalitlar)
        return {"success": True}
    return {"success": False, "error": "Kalitlar uzunligi 25 ta bo'lishi kerak"}

@app.post("/tekshirish")
async def tekshirish(file: UploadFile = File(...), student: str = Form("Noma'lum")):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    rasm = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if rasm is None:
        return JSONResponse({"success": False, "error": "Rasm yuklanmadi"})

    balandlik, kenglik = rasm.shape[:2]
    if balandlik > 1500:
        nisbat = 1200 / balandlik
        rasm = cv2.resize(rasm, (int(kenglik * nisbat), 1200))

    tekis = markerni_topish_va_tekislash(rasm)
    if tekis is None:
        return JSONResponse({"success": False, "error": "Varaqning 4 ta burchagi topilmadi. To'g'riroq qilib rasmga oling."})

    kalitlar = kalitlarni_olish()
    javoblar, ball, vizual_rasm = javoblarni_aniqlash(tekis, kalitlar)

    natijalar = {}
    for savol, javob in javoblar.items():
        k = kalitlar.get(savol, "-")
        holat = "To'g'ri" if javob == k else (f"Xato ({k})" if javob != "Belgilanmagan" else "-")
        natijalar[savol] = {"belgilangan": javob, "holat": holat}

    foiz = round((ball / 25) * 100, 1)

    # Rasm kodlash (telefonda ko'rsatish uchun)
    _, buffer = cv2.imencode('.jpg', vizual_rasm)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    # Baza tarixiga qo'shish
    baza_natijalar.append({
        "student": student,
        "ball": ball,
        "foiz": foiz,
        "vaqt": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "javoblar": {str(k): v for k, v in javoblar.items()}
    })

    return {
        "success": True,
        "student": student,
        "ball": ball,
        "foiz": foiz,
        "javoblar": natijalar,
        "image_base64": img_b64,
        "tarix": baza_natijalar
    }

@app.get("/download-excel")
def download_excel():
    wb = Workbook()
    ws = wb.active
    ws.title = "Natijalar"

    headers = ["#", "F.I.SH / ID", "To'g'ri javoblar", "Foiz (%)", "Sana/Vaqt"]
    for q in range(1, 26):
        headers.append(f"{q}-savol")
    ws.append(headers)

    for idx, item in enumerate(baza_natijalar, 1):
        row = [idx, item["student"], item["ball"], f"{item['foiz']}%", item["vaqt"]]
        for q in range(1, 26):
            row.append(item["javoblar"].get(str(q), "-"))
        ws.append(row)

    stream = io.BytesIO()
    wb.save(stream)
    stream.seek(0)

    filename = f"OMR_Natijalar_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    return StreamingResponse(
        stream,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)