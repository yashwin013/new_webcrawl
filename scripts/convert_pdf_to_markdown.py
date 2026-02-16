"""
Standalone script to convert PDF files to Markdown using Docling directly.

Bypasses AsyncDocumentProcessor to avoid silent GPU crashes in executor threads.
Uses DocumentConverter synchronously for better error visibility.

Usage:
    python scripts/convert_pdf_to_markdown.py <pdf_path>
    python scripts/convert_pdf_to_markdown.py --folder uploads/
    python scripts/convert_pdf_to_markdown.py --all
    python scripts/convert_pdf_to_markdown.py <pdf_path> --enable-ocr
"""

import sys
import os
import signal
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Force flush on print so we see output before any crash
import functools
print = functools.partial(print, flush=True)


def convert_pdf_to_markdown(pdf_path: Path, output_dir: Path = None, enable_ocr: bool = False) -> Path:
    """
    Convert a single PDF to markdown using Docling directly.
    No async, no executors — runs synchronously for reliability.
    """
    if output_dir is None:
        output_dir = Path("outputs/markdown")

    output_dir.mkdir(parents=True, exist_ok=True)

    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    print(f"PDF file: {pdf_path.name}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   OCR: {'Enabled' if enable_ocr else 'Disabled (use --enable-ocr to enable)'}")

    try:
        print("Step 1/4: Importing Docling...")
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
        print("   Done.")

        print("Step 2/4: Configuring pipeline...")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = enable_ocr
        pipeline_options.do_table_structure = True
        pipeline_options.do_code_enrichment = False
        pipeline_options.do_formula_enrichment = False
        pipeline_options.generate_page_images = False
        pipeline_options.images_scale = 1.0

        pipeline_options.allow_external_plugins = True

        if enable_ocr:
            try:
                from docling.datamodel.pipeline_options import (
                    AcceleratorDevice,
                    AcceleratorOptions,
                )
                from docling_surya import SuryaOcrOptions
                pipeline_options.ocr_options = SuryaOcrOptions(
                    force_full_page_ocr=True,
                )
                pipeline_options.accelerator_options = AcceleratorOptions(
                    device=AcceleratorDevice.CUDA,
                    num_threads=4,
                )
                print("   OCR: Surya with CUDA GPU")
            except ImportError:
                from docling.datamodel.pipeline_options import EasyOcrOptions
                pipeline_options.ocr_options = EasyOcrOptions(
                    force_full_page_ocr=True,
                    use_gpu=True,
                )
                print("   OCR: EasyOCR (fallback, docling-surya not installed)")

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )
        print("   Done.")

        print("Step 3/4: Converting PDF (this may take a while)...")
        result = converter.convert(source=str(pdf_path))
        print("   Conversion complete!")

        # Access the document from the result
        doc = result.document
        print(f"   Document type: {type(doc).__name__}")
        if hasattr(doc, 'pages'):
            print(f"   Pages: {len(doc.pages)}")

        print("Step 4/4: Exporting to markdown...")
        md_content = doc.export_to_markdown()

        if not md_content or len(md_content.strip()) == 0:
            print("   Markdown empty, trying text export as fallback...")
            if hasattr(doc, 'export_to_text'):
                md_content = doc.export_to_text()

        if not md_content or len(md_content.strip()) == 0:
            print("   ERROR: No content extracted from PDF.")
            print("   The PDF may be image-only. Try with --enable-ocr")
            return None

        # Save markdown file
        markdown_path = output_dir / f"{pdf_path.stem}.md"
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        print(f"\nSUCCESS: {markdown_path}")
        print(f"   Characters: {len(md_content):,}")
        print(f"   Lines: {md_content.count(chr(10)):,}")
        return markdown_path

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return None
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        return None


def convert_folder(folder_path: Path, output_dir: Path = None, enable_ocr: bool = False):
    """Convert all PDFs in a folder to markdown."""
    pdf_files = sorted(folder_path.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {folder_path}")
        return

    print(f"Found {len(pdf_files)} PDF files in: {folder_path}")
    print("=" * 60)

    successful = 0
    failed = 0

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}]")
        result = convert_pdf_to_markdown(pdf_path, output_dir, enable_ocr)
        if result:
            successful += 1
        else:
            failed += 1
        print()

    print("=" * 60)
    print(f"Successful: {successful}/{len(pdf_files)}")
    if failed:
        print(f"Failed: {failed}/{len(pdf_files)}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDF files to Markdown using Docling")
    parser.add_argument("pdf_path", nargs="?", help="Path to PDF file")
    parser.add_argument("--folder", help="Process all PDFs in folder")
    parser.add_argument("--all", action="store_true", help="Process all PDFs in uploads/")
    parser.add_argument("--output", help="Output directory (default: outputs/markdown)")
    parser.add_argument("--enable-ocr", action="store_true",
                        help="Enable OCR for image-based PDFs (slower, uses GPU)")

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else Path("outputs/markdown")
    enable_ocr = args.enable_ocr

    if args.all:
        uploads_dir = Path("uploads")
        if not uploads_dir.exists():
            print(f"Uploads directory not found: {uploads_dir}")
            return
        convert_folder(uploads_dir, output_dir, enable_ocr)

    elif args.folder:
        folder = Path(args.folder)
        if not folder.exists():
            print(f"Folder not found: {folder}")
            return
        convert_folder(folder, output_dir, enable_ocr)

    elif args.pdf_path:
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            print(f"File not found: {pdf_path}")
            return
        convert_pdf_to_markdown(pdf_path, output_dir, enable_ocr)

    else:
        print("Please specify a PDF file, --folder, or --all")
        parser.print_help()


if __name__ == "__main__":
    main()
