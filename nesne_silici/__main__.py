"""Başlatıcı: ``python -m nesne_silici [dosya]``"""
from __future__ import annotations

import sys

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from .arayuz.ana_pencere import AnaPencere


def koyu_tema(app: QApplication) -> None:
    """Fusion stili + koyu palet: görüntü düzenleyicilerde göz yormayan, tutarlı görünüm."""
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(43, 44, 49))
    p.setColor(QPalette.ColorRole.WindowText, QColor(230, 230, 234))
    p.setColor(QPalette.ColorRole.Base, QColor(30, 31, 34))
    p.setColor(QPalette.ColorRole.AlternateBase, QColor(43, 44, 49))
    p.setColor(QPalette.ColorRole.Text, QColor(230, 230, 234))
    p.setColor(QPalette.ColorRole.Button, QColor(58, 59, 66))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(230, 230, 234))
    p.setColor(QPalette.ColorRole.ToolTipBase, QColor(58, 59, 66))
    p.setColor(QPalette.ColorRole.ToolTipText, QColor(230, 230, 234))
    p.setColor(QPalette.ColorRole.Highlight, QColor(255, 140, 0))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(20, 20, 20))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(120, 120, 126))
    p.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(120, 120, 126))
    app.setPalette(p)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    app.setApplicationName("Nesne Silici")
    koyu_tema(app)
    pencere = AnaPencere()
    pencere.show()
    if len(argv) > 1:
        pencere.ac_dosya(argv[1])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
