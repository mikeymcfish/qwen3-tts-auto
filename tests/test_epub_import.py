import tempfile
import unittest
import zipfile
from pathlib import Path

from audiobook_qwen3 import read_text_file, split_into_paragraphs


class EpubImportTests(unittest.TestCase):
    def test_epub_import_cleans_notes_images_and_punctuates_headings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            epub_path = Path(tmp_dir) / "sample.epub"
            self._write_sample_epub(epub_path)

            text = read_text_file(epub_path)
            paragraphs = split_into_paragraphs(text)

            self.assertIn("Chapter One.", paragraphs)
            self.assertIn("A paragraph before an image.", paragraphs)
            self.assertIn("Paragraph with note marker and more text.", paragraphs)
            self.assertIn("Chapter Two:", paragraphs)
            self.assertIn("Second chapter body paragraph.", paragraphs)

            self.assertNotIn("Table of Contents", text)
            self.assertNotIn("Figure caption should be removed.", text)
            self.assertNotIn("Footnote text should be removed.", text)

    def _write_sample_epub(self, target_path: Path) -> None:
        container_xml = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>
"""
        content_opf = """<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">test-book</dc:identifier>
    <dc:title>Test Book</dc:title>
  </metadata>
  <manifest>
    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
    <item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="ch2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="nav"/>
    <itemref idref="ch1"/>
    <itemref idref="ch2"/>
  </spine>
</package>
"""
        nav_xhtml = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body>
    <nav>
      <h1>Table of Contents</h1>
      <ol><li>Chapter One</li></ol>
    </nav>
  </body>
</html>
"""
        ch1_xhtml = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
  <body>
    <section>
      <h1>Chapter One</h1>
      <p>A paragraph before an image.</p>
      <figure>
        <img src="cover.jpg" alt="cover"/>
        <figcaption>Figure caption should be removed.</figcaption>
      </figure>
      <p>Paragraph with note<a epub:type="noteref" href="#fn1">1</a> marker and more text.</p>
      <aside id="fn1" epub:type="footnote">
        <p>Footnote text should be removed.</p>
      </aside>
    </section>
  </body>
</html>
"""
        ch2_xhtml = """<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
  <body>
    <h2>Chapter Two:</h2>
    <p>Second chapter body paragraph.</p>
  </body>
</html>
"""

        with zipfile.ZipFile(target_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("mimetype", "application/epub+zip")
            archive.writestr("META-INF/container.xml", container_xml)
            archive.writestr("OEBPS/content.opf", content_opf)
            archive.writestr("OEBPS/nav.xhtml", nav_xhtml)
            archive.writestr("OEBPS/ch1.xhtml", ch1_xhtml)
            archive.writestr("OEBPS/ch2.xhtml", ch2_xhtml)


if __name__ == "__main__":
    unittest.main()
