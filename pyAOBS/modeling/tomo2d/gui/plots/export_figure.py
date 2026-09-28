"""绘图保存：PNG/JPEG/TIFF/BMP、PDF、PS/EPS、SVG（文件框允许模态）。"""

from __future__ import annotations

import warnings
from pathlib import Path

from PySide6.QtWidgets import QMessageBox, QWidget

SAVE_IMAGE_FILTER = (
    "PNG (*.png);;"
    "JPEG (*.jpg *.jpeg);;"
    "TIFF (*.tif *.tiff);;"
    "BMP (*.bmp);;"
    "PDF (*.pdf);;"
    "PostScript (*.ps);;"
    "Encapsulated PS (*.eps);;"
    "SVG (*.svg);;"
    "所有文件 (*)"
)

_RASTER = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_SVG = {".svg"}
_PAGE = {".pdf", ".ps", ".eps"}

# ISO A4（英寸）。PS 默认纸面是 Letter 竖版（宽 8.5"），宽剖面会被裁掉。
_A4_INCH = (8.27, 11.69)
_PS_MARGIN_INCH = 0.45


def _fail(msg: str) -> None:
    from ..dialog_utils import show_modeless_message

    show_modeless_message("保存失败", msg, icon=QMessageBox.Icon.Warning)


def _export_item(widget: QWidget):
    ci = getattr(widget, "ci", None)
    if ci is not None:
        return ci
    getter = getattr(widget, "getPlotItem", None)
    if callable(getter):
        return getter()
    return None


def _qimage_from_widget(widget: QWidget, *, scale: float = 2.0):
    from PySide6.QtGui import QColor, QImage
    from PySide6.QtWidgets import QApplication

    item = _export_item(widget)
    if item is not None:
        try:
            from pyqtgraph.exporters import ImageExporter

            exp = ImageExporter(item)
            w = max(int(float(exp.params["width"]) * scale), 1)
            h = max(int(float(exp.params["height"]) * scale), 1)
            exp.params["width"] = w
            exp.params["height"] = h
            try:
                exp.params["background"] = QColor(255, 255, 255)
            except Exception:
                pass
            img = exp.export(toBytes=True)
            if isinstance(img, QImage) and not img.isNull():
                return img
        except Exception:
            pass
    QApplication.processEvents()
    pix = widget.grab()
    if pix.isNull():
        return None
    return pix.toImage()


def _save_svg(widget: QWidget, path: str) -> None:
    from PySide6.QtGui import QColor
    from pyqtgraph.exporters import SVGExporter

    item = _export_item(widget)
    if item is None:
        raise RuntimeError("当前图不支持 SVG 矢量导出，请改用 PNG/PDF")
    exp = SVGExporter(item)
    try:
        exp.params["background"] = QColor(255, 255, 255)
    except Exception:
        pass
    exp.export(path)


def _save_pdf_from_qimage(img, path: str, *, dpi: float = 96.0) -> None:
    """把位图写入 PDF。刻度数字随像素走，避免 Qt PDF 嵌入中文字体时拉丁数字乱码。"""
    from PySide6.QtCore import QMarginsF, QSizeF
    from PySide6.QtGui import QColor, QPageLayout, QPageSize, QPainter, QPdfWriter

    w = max(int(img.width()), 1)
    h = max(int(img.height()), 1)
    dpi = float(dpi) if dpi and dpi > 0 else 96.0
    writer = QPdfWriter(path)
    writer.setResolution(int(round(dpi)))
    writer.setPageSize(
        QPageSize(QSizeF(w * 25.4 / dpi, h * 25.4 / dpi), QPageSize.Unit.Millimeter)
    )
    writer.setPageMargins(QMarginsF(0.0, 0.0, 0.0, 0.0), QPageLayout.Unit.Millimeter)
    painter = QPainter(writer)
    try:
        target = painter.viewport()
        painter.fillRect(target, QColor(255, 255, 255))
        painter.drawImage(target, img)
    finally:
        painter.end()


def _save_pdf_scene(widget: QWidget, path: str) -> None:
    """pyqtgraph → PDF：截图嵌入，不用 QPrinter 矢量描场景。

    HighResolution QPrinter 会把场景像素当成印刷点、再用 1200 dpi 去画文字；
    轴刻度又常落在微软雅黑上，PDF 里数字子集编码对不上，看起来就是乱码。
    """
    scale = 2.0
    img = _qimage_from_widget(widget, scale=scale)
    if img is None or img.isNull():
        raise RuntimeError("无法抓取图像")
    _save_pdf_from_qimage(img, path, dpi=96.0 * scale)


def _fit_figsize_to_a4(width_in: float, height_in: float) -> tuple[float, float, str]:
    """把英寸尺寸缩进 A4 可印区域；宽≥高用横向。返回 (w, h, orientation)。"""
    w = max(float(width_in), 1e-3)
    h = max(float(height_in), 1e-3)
    landscape = w >= h
    page_w, page_h = (_A4_INCH[1], _A4_INCH[0]) if landscape else _A4_INCH
    max_w = max(page_w - 2 * _PS_MARGIN_INCH, 1.0)
    max_h = max(page_h - 2 * _PS_MARGIN_INCH, 1.0)
    scale = min(max_w / w, max_h / h, 1.0)
    return w * scale, h * scale, "landscape" if landscape else "portrait"


def _ps_savefig_kwargs(orientation: str) -> dict:
    return {
        "dpi": 300,
        "orientation": orientation,
        "papertype": "a4",
        "facecolor": "white",
        "edgecolor": "none",
    }


def _flatten_artist_alphas(fig) -> list[tuple[object, object]]:
    """PS 不支持半透明：把 alpha<1 的 artist 临时设为不透明。"""
    changed: list[tuple[object, object]] = []
    find = getattr(fig, "findobj", None)
    if not callable(find):
        return changed
    for artist in find():
        if not hasattr(artist, "get_alpha") or not hasattr(artist, "set_alpha"):
            continue
        try:
            old = artist.get_alpha()
        except Exception:
            continue
        if old is None:
            continue
        try:
            if float(old) >= 1.0:
                continue
        except (TypeError, ValueError):
            pass
        try:
            artist.set_alpha(1.0)
            changed.append((artist, old))
        except Exception:
            continue
    return changed


def _restore_artist_alphas(changed: list[tuple[object, object]]) -> None:
    for artist, old in changed:
        try:
            artist.set_alpha(old)
        except Exception:
            pass


def _savefig_ps(fig, path: str, fmt: str, *, tight: bool) -> None:
    orig = tuple(float(x) for x in fig.get_size_inches())
    fw, fh, orientation = _fit_figsize_to_a4(orig[0], orig[1])
    resized = abs(fw - orig[0]) > 0.02 or abs(fh - orig[1]) > 0.02
    if resized:
        fig.set_size_inches(fw, fh, forward=False)
    kw = _ps_savefig_kwargs(orientation)
    kw["format"] = fmt.lstrip(".")
    if tight:
        kw["bbox_inches"] = "tight"
        kw["pad_inches"] = 0.12
    else:
        kw["pad_inches"] = 0
    changed = _flatten_artist_alphas(fig)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The PostScript backend does not support transparency.*",
            )
            try:
                fig.savefig(path, **kw)
            except TypeError:
                kw.pop("papertype", None)
                fig.savefig(path, **kw)
    finally:
        _restore_artist_alphas(changed)
        if resized:
            fig.set_size_inches(orig[0], orig[1], forward=False)


def _save_page_from_qimage(img, path: str, fmt: str) -> None:
    from io import BytesIO

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice

    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    if not img.save(buf, "PNG"):
        raise RuntimeError("无法把图转成中间 PNG")
    buf.close()
    import matplotlib.image as mpimg

    arr = mpimg.imread(BytesIO(bytes(ba)))
    if arr.ndim == 3 and arr.shape[-1] == 4:
        rgb = arr[..., :3]
        a = arr[..., 3:4]
        arr = rgb * a + (1.0 - a)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    fig = Figure(figsize=(max(w, 1) / 100.0, max(h, 1) / 100.0), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(arr)
    ax.axis("off")
    _savefig_ps(fig, path, fmt, tight=False)
    fig.clear()


def _qt_raster_format(suffix: str) -> str:
    return {
        ".png": "PNG",
        ".jpg": "JPG",
        ".jpeg": "JPG",
        ".bmp": "BMP",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".webp": "WEBP",
    }.get(suffix, "PNG")


def save_graphics_widget(
    parent: QWidget,
    widget: QWidget,
    *,
    start_dir: str = "",
    default_name: str = "plot.png",
) -> None:
    """弹出保存框，把 pyqtgraph GraphicsLayoutWidget / PlotWidget 写成多种格式。"""
    from pyAOBS.utils.qt_file_dialog import get_save_file_name

    stem = Path(default_name).stem or "plot"
    start = str(Path(start_dir) / stem) if start_dir else stem
    path, _sel = get_save_file_name(
        parent,
        "保存图像",
        start,
        SAVE_IMAGE_FILTER,
        default_suffix=".png",
    )
    if not path:
        return
    suffix = Path(path).suffix.lower()
    try:
        if suffix in _SVG:
            _save_svg(widget, path)
            return
        if suffix == ".pdf":
            _save_pdf_scene(widget, path)
            return
        if suffix in {".ps", ".eps"}:
            img = _qimage_from_widget(widget)
            if img is None or img.isNull():
                raise RuntimeError("无法抓取图像")
            _save_page_from_qimage(img, path, suffix.lstrip("."))
            return
        if suffix not in _RASTER:
            path = path + ".png"
            suffix = ".png"
        img = _qimage_from_widget(widget)
        if img is None or img.isNull():
            raise RuntimeError("无法抓取图像")
        qfmt = _qt_raster_format(suffix)
        quality = 92 if qfmt == "JPG" else -1
        if not img.save(path, qfmt, quality):
            raise RuntimeError(f"无法写入: {path}")
    except Exception as e:
        _fail(str(e))


def save_mpl_figure(
    parent: QWidget,
    fig,
    *,
    start_dir: str = "",
    default_name: str = "plot.png",
) -> None:
    """Matplotlib Figure：savefig 直接支持 PNG/JPG/PDF/PS/EPS/SVG/TIFF。"""
    from pyAOBS.utils.qt_file_dialog import get_save_file_name

    stem = Path(default_name).stem or "plot"
    start = str(Path(start_dir) / stem) if start_dir else stem
    path, _sel = get_save_file_name(
        parent,
        "保存图像",
        start,
        SAVE_IMAGE_FILTER,
        default_suffix=".png",
    )
    if not path:
        return
    suffix = Path(path).suffix.lower()
    if suffix not in _RASTER | _SVG | _PAGE:
        path = path + ".png"
        suffix = ".png"
    try:
        if suffix in {".ps", ".eps"}:
            _savefig_ps(fig, path, suffix, tight=True)
        else:
            fig.savefig(path, dpi=300, bbox_inches="tight")
    except Exception as e:
        _fail(str(e))
