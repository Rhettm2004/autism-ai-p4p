import 'dart:typed_data';

import 'report_download_stub.dart'
    if (dart.library.js_interop) 'report_download_web.dart';

Future<void> downloadReportPdf({
  required Uint8List bytes,
  required String filename,
}) {
  return downloadReportPdfImpl(bytes: bytes, filename: filename);
}
