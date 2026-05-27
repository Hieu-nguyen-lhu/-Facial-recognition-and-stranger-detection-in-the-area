# -*- mode: python ; coding: utf-8 -*-

import cv2


a = Analysis(
    ['admin_app.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        ('models', 'models'),
        ('yolov8n.pt', '.'),
        ('known_faces', 'known_faces'),
        (cv2.data.haarcascades, 'cv2/data'),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FaceSecurity',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['assets\\app_icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FaceSecurity',
)
