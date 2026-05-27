# Facial recognition and stranger detection in the area

De tai Python cho linh vuc an ninh va giam sat: dung camera/video, YOLO de phat hien nguoi trong khung hinh, sau do nhan dien khuon mat tu thu muc nguoi quen. Neu khong khop voi du lieu da dang ky, he thong danh dau la nguoi la, luu anh va ghi log canh bao.

## Chuc nang

- Phat hien nguoi bang YOLO (`ultralytics`).
- Nhan dien khuon mat tu anh da dang ky trong `known_faces/`.
- Uu tien OpenCV LBPH (`opencv-contrib-python`) de nhan dien chinh xac hon fallback histogram/ORB cu.
- Co man hinh quan ly nguoi quen: them tu anh, chup bang camera, xoa va xem lai danh sach.
- Tu dong dung `face_recognition` neu may da cai duoc, fallback sang OpenCV neu chua co.
- Ve khung xanh cho nguoi quen, khung do cho nguoi la.
- Luu anh nguoi la vao `captures/`.
- Ghi lich su canh bao vao `logs/detections.csv`.
- Chi chup/canh bao nguoi la sau khi phat hien lien tuc 10 giay.
- Ho tro webcam, camera IP hoac file video.

## Cai dat

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
```

Neu `py -m venv .venv` loi o buoc `ensurepip`, co the cai truc tiep vao Python hien tai:

```powershell
py -m pip install -r requirements.txt
```

Neu muon backend nhan dien tot hon va may cai duoc `dlib`, co the cai them:

```powershell
py -m pip install face_recognition
```

## Quan ly nguoi quen

Mo man hinh quan ly:

```powershell
py admin_app.py
```

Trong man hinh nay co the:

- Them nguoi tu 3 file anh: chinh dien, xoay trai, xoay phai.
- Chup 3 goc khuon mat bang webcam theo kieu tu quet: dua mat vao khung, giu yen, he thong tu chup nhieu mau cho moi goc.
- Xem danh sach nguoi da dang ky.
- Xoa nguoi khoi du lieu.

Du lieu se duoc luu theo tung thu muc rieng trong `known_faces/`, nen khi chay `main.py` he thong se doc lai danh sach moi.

Vi du:

```text
known_faces/
+-- Nguyen_Van_A/
    +-- front.jpg
    +-- front_2.jpg
    +-- left.jpg
    +-- left_2.jpg
    +-- right.jpg
    +-- right_2.jpg
```

## Dang ky bang lenh

Dang ky tu file anh:

```powershell
py register_face.py --name "Nguyen Van A" --image path\to\avatar.jpg
```

Dang ky tu webcam:

```powershell
py register_face.py --name "Nguyen Van A" --camera 0
```

## Chay giam sat

Chay bang webcam mac dinh:

```powershell
py main.py
```

Mac dinh anh nguoi la chi duoc luu sau khi nguoi la xuat hien lien tuc 10 giay. Co the doi so giay:

```powershell
py main.py --stranger-hold-seconds 15
```

Camera se hien thi moi frame, con YOLO/nhan dien chay o luong nen de bot giat man hinh. Mac dinh van gui moi frame cho bo nhan dien de giu kha nang nhan nguoi quen.

Neu may van qua lag, co the giam tan suat chay YOLO/nhan dien:

```powershell
py main.py --process-every 3
```

Gia tri nho hon se nhan dien nguoi quen tot va nhanh hon, gia tri lon hon se muot hon nhung ket qua nhan dien cap nhat cham hon. Mac dinh la `1`.

Neu nguoi quen van kho duoc xac nhan, co the giam so frame can bo phieu:

```powershell
py main.py --vote-window 5 --vote-min 3
```

Cach nay de nhan nguoi quen hon, nhung neu moi truong co nhieu nguoi la thi nen kiem tra ky de tranh nhan nham.

Neu he thong con nhan nham nguoi la thanh nguoi quen, giam nguong LBPH:

```powershell
py main.py --lbph-threshold 60
```

Neu he thong qua chat va hay bao nguoi quen la nguoi la, tang nhe nguong:

```powershell
py main.py --lbph-threshold 75
```

Chay voi file video:

```powershell
py main.py --source path\to\video.mp4
```

Chay voi camera IP/RTSP:

```powershell
py main.py --source rtsp://user:password@camera-ip:554/stream
```

Phim tat khi cua so camera dang mo:

- `q`: thoat chuong trinh
- `s`: luu nhanh anh khung hinh hien tai vao `captures/`

## Cau truc thu muc

```text
.
+-- admin_app.py
+-- main.py
+-- register_face.py
+-- requirements.txt
+-- known_faces/
+-- captures/
+-- logs/
+-- src/
    +-- alert_logger.py
    +-- config.py
    +-- face_recognizer.py
    +-- face_store.py
    +-- yolo_detector.py
```

## Ghi chu model

Mac dinh chuong trinh dung `yolov8n.pt`, model nhe co san cua Ultralytics de phat hien `person`. Lan chay dau tien Ultralytics co the tu tai model nay neu may co Internet.

Neu ban co model YOLO rieng cho face detection, co the truyen vao:

```powershell
py main.py --model models\your-face-or-person-model.pt
```
